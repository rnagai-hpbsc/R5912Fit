#!/usr/bin/env python3

import tables
import os
import sys
import numpy as np
from ROOT import *
from argparse import ArgumentParser
from tqdm import tqdm
from scipy.special import comb

def main():
    args = parser()

    filename = args.filename

    os.system('mkdir -p rootfiles')
    os.system('mkdir -p plots/paper')

    f = tables.open_file(filename)

    data = f.get_node('/data')
    event_ids = data.col('event_id')
    times = data.col('time')
    waveforms = data.col('waveform')

    timestamps = getDataCol('timestamp',data)
    hv = getDataCol('hv',data)
    temp = getDataCol('temperature',data)

    nsamples = len(waveforms[0])
    fq = np.linspace(0,240,nsamples)

    gmean = np.mean(waveforms[0])
    globalmean = np.mean(waveforms)
    print(np.mean(waveforms[0]))

    ofilename = filename.split('/')[len(filename.split('/'))-1].split('.h')[0] + args.ofile

    of = TFile(f'rootfiles/{ofilename}.root',"RECREATE")

    h     = TH1D('qdist','qdist;ADC count [LSB];Entry',2500,-100,4900)
    hmaxbin  = TH1D('maxbin','maxbin;ADC count [LSB];Entry', nsamples, 0, nsamples)
    hopt  = TH1D('optq','optq;ADC count [LSB];Entry', 2500, -100,4900)
    hoptq = TH1D('optqcal','optqcal;Charge [pC];Entry', 5000, -1,49)
    hpk   = TH1D('peak','peak;ADC count [LSB];Entry',500,-100,900)
    h1e7pk   = TH1D('1e7peak','1e7peak;ADC count [LSB];Entry',1000,-100,900)
    havg  = TH1D('avgwf','Averaged Waveform;Sampling Bin;ADC count',nsamples,0,nsamples)
    gHV = TGraph()
    gHV.SetTitle('gHV')
    gHV.SetName('gHV')
    for i in range(len(hv)): 
        n = gHV.GetN()
        gHV.Set(n+1)
        gHV.SetPoint(n,i,hv[i])
    gTemp = TGraph()
    gTemp.SetTitle('gTemp')
    gTemp.SetName('gTemp')
    for i in range(len(temp)): 
        n = gTemp.GetN()
        gTemp.Set(n+1)
        gTemp.SetPoint(n,i,temp[i])
    gPkInt = TGraph()
    gPkInt.SetTitle('gPkInt')
    gPkInt.SetName('gPkInt')
    
    winmin = args.minimum
    winmax = args.minimum + args.window
    bsstart = args.baselineEst

    print (f'Total #Events: {len(waveforms)}')
    print (f'Setting... Window [{winmin}:{winmax}] and Pedestal start from {bsstart}.\n')

    topdir = gDirectory.GetDirectory(gDirectory.GetPath())
    subdir = topdir.mkdir("Waveforms")
    subtdir = topdir.mkdir("SubtWaveforms")

    fltwfs = []
    starttimestamp = timestamps[0]

    Mbcalib = 0.075e-3 # V/LSB for Rev4
    if args.Rev3: 
        Mbcalib = 0.0787e-3 #V/LSB
    Mbspfq = 240.e6 # Hz sampling freq
    Mbimped = 50. # ohm impedance
    elecQ = 1.60217662e-19 # elementary charge

    pulseheights = []
    for i in tqdm(range(len(waveforms))):
        waveform = waveforms[i]
        timestamp = timestamps[i]

        baseline_mean = np.mean(waveform[160:])
        selected = waveform[waveform < np.mean(waveform) + 10]
        reduced_waveform = waveform - baseline_mean
        scale = (nsamples-bsstart)/(winmax-winmin)
        h.Fill(sum(reduced_waveform[winmin:winmax]))

        maxbin = np.argmax(waveform)
        hmaxbin.Fill(maxbin)
        if args.laser: 
            maxbin = args.laser
        elif maxbin < 50: 
            maxbin = 50
        elif maxbin > nsamples - 20: 
            maxbin = nsamples - 20
        hopt.Fill(sum(waveform[maxbin-10:maxbin+10])-20*np.mean(waveform[:maxbin-20]))
        hoptq.Fill(sum(reduced_waveform[maxbin-10:maxbin+10])*Mbcalib/Mbimped/Mbspfq*1.e12)

        pulseheights.append(max(waveform[winmin:winmax])-np.mean(waveform[160:]))
        hpk.Fill(max(waveform[winmin:winmax])-np.mean(waveform[bsstart:nsamples]))
        h1e7pk.Fill((max(waveform[winmin:winmax])-np.mean(waveform[bsstart:nsamples]))/1.1)

        n = gPkInt.GetN()
        gPkInt.Set(n+1)
        gPkInt.SetPoint(n,sum(waveform[maxbin-6:maxbin+8])-14*np.mean(waveform[160:]),max(waveform[winmin:winmax])-np.mean(waveform[160:]))
        if max(waveform) - np.mean(waveform[bsstart:nsamples]) < args.threshold:
            continue

        fltwfs.append(waveform)

        if args.noeachwf: 
            h2 = TH1D(f'w{i}',f'Waveform{i};Sampling Bin;ADC count [LSB]',nsamples,0,nsamples)
            for j in range(len(waveform)):
                h2.Fill(j,waveform[j])
            subdir.cd()
            h2.Write()
            h2 = TH1D(f'sw{i}',f'SubtWaveform{i};Sampling Bin;ADC count [LSB]',nsamples,0,nsamples)
            for j in range(len(waveform)):
                h2.Fill(j,waveform[j]-baseline_mean)
            subtdir.cd()
            h2.Write()

    print('')

    if args.compwf: 
        savePulseHeightHist(pulseheights,waveforms)

    avgfltwfs = np.mean(fltwfs, axis=0)
    for i in range(len(avgfltwfs)): 
        havg.Fill(i,avgfltwfs[i])

    topdir.cd()
    gHV.Write()
    gTemp.Write()
    gPkInt.Write()
    h.Write()
    hopt.Write()
    hoptq.Write()
    hpk.Write()
    h1e7pk.Write()
    havg.Write()
    hmaxbin.Write()

    rebin = args.rebin
    hopt.Rebin(rebin)
    hsubt = hopt.Clone()
    hopt.Fit('gausn',"","",-20,20)
        
    fped = hopt.GetFunction('gausn')
    ped, pederr = get3Parameters(fped)
    fped = TF1("ped","gausn",-100,1900)
    fped.SetParameters(ped[0],ped[1],ped[2])
    xtrans = ped[1]
    #pmean = -1. * np.log(ped[0]/2./rebin/hopt.GetEntries()) #0.3 # poisson mean
    pmean = -1. * np.log(fped.Integral(-100,1900)/2./rebin/hopt.GetEntries()) #0.3 # poisson mean
    qreso = 0.3 # charge resolution 
    qdaq = ped[2] #0.07 # DAQ resolution 
    loss1 = 0.07 # 7% loss @ 1st dynode
    print(f'Poisson Mean: {pmean}')

    fitmax = 380
    hopt.GetXaxis().SetRangeUser(xtrans+qdaq*5,4900) 
    maxfitrange = hopt.GetBinCenter(hopt.GetMaximumBin())
    hopt.GetXaxis().UnZoom()
    interval = maxfitrange - (xtrans+qdaq*5)
    if interval < 1: 
        interval = 5
    print(f'Est SPE Mean: {maxfitrange}, Interval: {interval}')

    hopt.Fit('gaus',"","",xtrans+qdaq*5,maxfitrange+interval)
    est1mean = hopt.GetFunction('gaus').GetParameter(1)
    estsigma = hopt.GetFunction('gaus').GetParameter(2)
    lowlimit = qdaq*5 
    if lowlimit < est1mean/2.: 
        lowlimit = est1mean/2.
    #hopt.Fit('gaus',"","",lowlimit,est1mean*1.5)
    if est1mean-estsigma/2. < est1mean/2.:
        lowlimit = est1mean/2.
    else: 
        lowlimit = est1mean-estsigma/2.
    hopt.Fit('gaus',"","",lowlimit,est1mean+estsigma/2.)
    fspe = hopt.GetFunction('gaus')
    spe, speerr = get3Parameters(fspe)
    fspe = TF1("spe","gaus",-100,1900)
    fspe.SetParameters(spe[0],spe[1],spe[2])
    qreso = np.sqrt(spe[2]**2-qdaq**2)/(spe[1]-xtrans)
    Ratio = fspe.Integral(-100,1900)/fped.Integral(-100,1900)
    print(f'Poisson Mean: {pmean}, Ratio: {fspe.Integral(-100,1900)/fped.Integral(-100,1900)}')

    k = 0.75
    mpe = args.mpe # number of p.e. 
    a = np.array([16.8, 4, 5, 3.33, 1.67, 1, 1.2, 1.5, 2.2, 3, 2.4]) # R5912-100 Divider list 
    Total = np.sum(a)
    R = a/Total # each divider Ratio 

    R2_10 = np.prod(R[1:10])
    A = 1./ (np.prod(R[0:10])**(k/10.))
    gain1 = A * R[0]**k
    rgain1  = R[0]/np.prod(R[0:10]**0.1)
    rgain2_ = R2_10/(np.prod(R[0:10])**0.9)
    rgain3_ = np.prod(R[2:10])/(np.prod(R[0:10])**0.8)

    R5912Model = "0"
    NPEs = []
    for i in range(mpe+1): 
        NPE = "0"
        for j in range(i+1): 
            # Model B 
            Qmean = f'({j}+{(i-j)}*pow({rgain2_},[3]))'
            NPE_j = f'+[0]*{comb(i,j)}*pow((1-[2]),{j})*pow([2],{i-j})*{NphotonDist(i,[4],Qmean,qreso,qdaq,xtrans)}'
            NPE += NPE_j
            R5912Model += NPE_j
        NPEs.append(TF1(f'f_{i}PE',NPE,-100,1900))

    f4 = TF1("R5912Model",R5912Model,-100.,1900.)
    f4.SetLineColor(2)
    f4.SetNpx(500)

    preQMean = spe[1]

    f4.SetParameter(1,preQMean)
    f4.SetParameter(2,loss1)
    f4.SetParLimits(2,0,0.5)
    f4.SetParameter(4,pmean)
    if args.poisfix:
        f4.SetParLimits(4,1,1)
    if args.k: 
        f4.SetParameter(3,float(args.k))
        f4.SetParLimits(3,0.75,0.75)
    else: 
        f4.SetParameter(3,0.75)
        f4.SetParLimits(3,0.5,0.9)
    hopt.Fit("R5912Model","","",-100,1900)
    pmtpar, pmterr = get3Parameters(f4,5)

    hsubt.SetName("hsubt")
    hsubt.SetTitle("hsubt;ADC count [LSB];Data - Fit")
    hsubt.Add(f4,-1)

    obsgain = f4.GetParameter(1)*Mbcalib/Mbimped/Mbspfq/elecQ

    print(f'{pmean}, {xtrans}, {qdaq}, {f4.GetParameter(0)}')
    print(f'Observed gain: {f4.GetParameter(1)*Mbcalib/Mbimped/Mbspfq/elecQ:.5e} @ {np.mean(hv):.1f} V')

    f4.Write()
    hsubt.Write()

    valley  = f4.GetMinimum(xtrans,preQMean)
    valleyx = f4.GetMinimumX(xtrans,preQMean)
    peak    = f4.GetMaximum(valleyx,1900)
    peakx   = f4.GetMaximumX(valleyx,1900)
    print(f'Peak:{peak:.1f} @ {peakx:.1f}, Valley:{valley:.1f} @ {valleyx:.1f}, P/V:{peak/valley:.3f}')

    hPars = TH1F("FitPars","FitPars",12,0,12)
    hPars.GetXaxis().SetBinLabel(1,"MB calib [mV/LSB]")
    hPars.SetBinContent(1,Mbcalib*1.e3)
    hPars.SetBinError(1,Mbcalib*10) # 1%
    hPars.GetXaxis().SetBinLabel(2,"MB Sampling Freq [MHz]")
    hPars.SetBinContent(2,Mbspfq/1.e6)
    hPars.SetBinError(2,0) # no error
    hPars.GetXaxis().SetBinLabel(3,"MB impedance [#Omega]")
    hPars.SetBinContent(3,Mbimped)
    hPars.SetBinError(3,5) # 10% 
    hPars.GetXaxis().SetBinLabel(4,"Rebin")
    hPars.SetBinContent(4,rebin)
    hPars.SetBinError(4,0) # no error
    hPars.GetXaxis().SetBinLabel(5,"Poisson Mean")
    hPars.SetBinContent(5,pmean)
    hPars.SetBinError(5, 1./ped[0]*pederr[0])
    hPars.GetXaxis().SetBinLabel(6,"Pedestal Mean [LSB]")
    hPars.SetBinContent(6,xtrans)
    hPars.SetBinError(6, pederr[1])
    hPars.GetXaxis().SetBinLabel(7,"Pedestal Sigma [LSB]")
    hPars.SetBinContent(7,qdaq)
    hPars.SetBinError(7, pederr[2])
    hPars.GetXaxis().SetBinLabel(8,"Total Gain [10^{7}]")
    hPars.SetBinContent(8,f4.GetParameter(1)*Mbcalib/Mbimped/Mbspfq/elecQ/1.e7)
    hPars.SetBinError(8,f4.GetParError(1)*Mbcalib/Mbimped/Mbspfq/elecQ/1.e7)
    hPars.GetXaxis().SetBinLabel(9,"Charge Resolution [LSB]")
    hPars.SetBinContent(9,qreso)
    hPars.SetBinError(9,np.sqrt(speerr[2]**2-pederr[2]**2)/(spe[1]-xtrans))
    hPars.GetXaxis().SetBinLabel(10,"Peak to Valley Ratio")
    hPars.SetBinContent(10,peak/valley)
    hPars.GetXaxis().SetBinLabel(11,"1st Dynode Loss Probability")
    hPars.SetBinContent(11,f4.GetParameter(2))
    hPars.SetBinError(11,f4.GetParError(2))
    hPars.GetXaxis().SetBinLabel(12,"Slope Constant of Gain Curve") 
    hPars.SetBinContent(12,f4.GetParameter(3))
    hPars.SetBinError(12,f4.GetParError(3))

    hPars.Write()

    c = TCanvas("c1","c1",800,600)
    c.Draw()
    c.SetLeftMargin(0.13)
    c.SetBottomMargin(0.12)
    hopt.SetMarkerStyle(20)
    hopt.SetMarkerSize(1)
    hopt.GetXaxis().SetRangeUser(-50,args.xmax)
    hopt.GetYaxis().SetRangeUser(0,(hopt.GetBinContent(hopt.GetXaxis().FindBin(est1mean))+hopt.GetBinError(hopt.GetXaxis().FindBin(est1mean)))*1.25)
    hopt.GetXaxis().SetTitleOffset(1.15)
    hopt.GetYaxis().SetTitleOffset(1.15)
    hopt.Draw("PE")

    leg = TLegend(.65,.5,.93,.93)
    leg.SetTextSize(0.04)
    leg.AddEntry(hopt,"Data","PE")
    leg.AddEntry(f4,"Fit","L")
    for i in range(mpe+1): 
        NPEs[i].SetParameters(pmtpar[0],pmtpar[1],pmtpar[2],pmtpar[3],pmtpar[4])
        NPEs[i].SetLineStyle(2+i)
        NPEs[i].SetLineColor(kAzure+i)
        NPEs[i].Draw("same")
        leg.AddEntry(NPEs[i],f"{i}PE contrib.","L")
    leg.Draw()
    c.SaveAs(f"plots/paper/{ofilename}.pdf")


    c = TCanvas("c2","c2",1000,800)
    c.Draw()
    rp = TRatioPlot(hopt)
    rp.SetH1DrawOpt("PE")
    rp.SetFitDrawOpt("L")
    rp.SetGraphDrawOpt("PE")
    rp.Draw()
    rp.GetLowerRefYaxis().SetTitle("Data/Fit")
    rp.GetUpperRefYaxis().SetTitle("Entries")
    rp.GetUpperRefYaxis().SetRangeUser(0,peak*1.5)
    rp.GetUpperRefXaxis().SetRangeUser(-50,args.xmax)
    rp.SetSeparationMargin(0.01)
    rp.SetRightMargin(0.04)
    rp.SetLeftMargin(0.12)
    rp.SetUpTopMargin(0.05)
    rp.SetLowBottomMargin(0.4)
    rp.GetLowerRefXaxis().SetTitleOffset(1.2)
    rp.GetLowerRefYaxis().SetTitleOffset(1.2)
    rp.GetUpperRefYaxis().SetTitleOffset(1.2)
    rp.GetLowerRefXaxis().SetTitleSize(0.04)
    rp.GetLowerRefYaxis().SetTitleSize(0.04)
    rp.GetUpperRefYaxis().SetTitleSize(0.04)
    rp.GetLowerRefXaxis().SetLabelSize(0.04)
    rp.GetLowerRefYaxis().SetLabelSize(0.04)
    rp.GetUpperRefYaxis().SetLabelSize(0.04)
    c.SetGrid()
    c.Update()
    c.SaveAs(f"plots/ratiotest_{ofilename}.pdf")

    c.Write()
    of.Close()
    f.close()

def savePulseHeightHist(pulseheights, waveforms):
    npphs = np.array(pulseheights)
    hilists = []
    hilists.append(np.where((npphs> 1990) & (npphs< 2010))[0])
    hilists.append(np.where((npphs> 2990) & (npphs< 3010))[0])
    hilists.append(np.where((npphs> 3990) & (npphs< 4010))[0])
    hilists.append(np.where((npphs> 4990) & (npphs< 5010))[0])
    hilists.append(np.where((npphs> 5950) & (npphs< 6050))[0])
    hilists.append(np.where((npphs> 6950) & (npphs< 7050))[0])
    hilists.append(np.where((npphs> 7950) & (npphs< 8050))[0])
    hilists.append(np.where((npphs> 8950) & (npphs< 9050))[0])
    hilists.append(np.where((npphs> 9900) & (npphs<10000))[0])
    hilists.append(np.where((npphs>10900) & (npphs<11000))[0])
    hilists.append(np.where((npphs>11900) & (npphs<12000))[0])
    hilists.append(np.where((npphs>12900) & (npphs<13000))[0])
    hilists.append(np.where((npphs>13900) & (npphs<14000))[0])
    hilists.append(np.where((npphs>14800)))

    print(hilists)
    
    c = TCanvas('c','c',800,600)
    c.SetLeftMargin(0.13)
    c.SetBottomMargin(0.12)
    c.SetGrid()
    c.Draw()

    hlists = []
    for i in range(len(hilists)): 
        for ii in range(len(hilists[i])): 
            print(type(hilists[i][ii]))
            if type(hilists[i][ii]) is np.ndarray: 
                continue
            waveform = waveforms[hilists[i][ii]]
            h = TH1D(f'w{hilists[i][ii]}',f'w{hilists[i][ii]};Sampling Bins; ADC Count [LSB]',len(waveform),0,len(waveform))
            for j in range(len(waveform)): 
                h.Fill(j,waveform[j])
            hlists.append(h)
            break

    leg = TLegend(.7,.3,.93,.93)
    for i in range(len(hlists)):
        hlists[i].SetLineColor(int(51+i*4))
        leg.AddEntry(hlists[i],f'Pulse height: {i*1000+2000}','L')
        if i==0: 
            tsize = 0.042
            hlists[i].GetXaxis().SetTitleSize(tsize)
            hlists[i].GetYaxis().SetTitleSize(tsize)
            hlists[i].GetXaxis().SetLabelSize(tsize)
            hlists[i].GetYaxis().SetLabelSize(tsize)
            hlists[i].GetXaxis().SetTitleOffset(1.15)
            hlists[i].GetYaxis().SetTitleOffset(1.5)
            hlists[i].GetXaxis().SetRangeUser(100,180)
            hlists[i].GetYaxis().SetRangeUser(0,17000)
            hlists[i].Draw("hist")
        else: 
            hlists[i].Draw("histsame")
        leg.Draw()

    c.SaveAs('pulseheight.pdf')


def NphotonDist(Num, PoisMean, QMean, QReso, Qdaq, xtrans=0): 
    return  f'TMath::PoissonI({Num},{PoisMean}) * TMath::Gaus(x-{xtrans}, [1]*{QMean}, TMath::Sqrt([1]*[1]*{QMean}*{QReso**2}+{Qdaq**2}),1)'
    '''
    for i in range(mpe+1): 
        for j in range(i+1): 
            # Model A
            #R5912Model += f'+[0]*{comb(i,j)*((1-loss1)**j)*(loss1**(i-j))}*{NphotonDist(i,pmean,j*gain1*gain2_10+(i-j)*gain2_10,qreso,qdaq,xtrans)}'
            # Model B: Standard
            Qmean = f'({j}+{(i-j)}*pow({rgain2_},[3]))'
            R5912Model += f'+[0]*{comb(i,j)}*pow((1-[2]),{j})*pow([2],{i-j})*{NphotonDist(i,pmean,Qmean,qreso,qdaq,xtrans)}'
            # Model C
            #Qmean = f'({j}*(1-[2]/2)+{(i-j)}*pow({rgain2_},[3]))'
            #R5912Model += f'+[0]*(1-[2]/2)*{comb(i,j)}*pow((1-[2]),{j})*pow([2],{i-j})*{NphotonDist(i,pmean,Qmean,qreso,qdaq,xtrans)}'
            #Qmean = f'({j}*[2]/2*pow({rgain1},[3])*pow({rgain3_},[3])+{(i-j)}*pow({rgain2_},[3]))'
            #R5912Model += f'+[0]*[2]/2*{comb(i,j)}*pow((1-[2]),{j})*pow([2],{i-j})*{NphotonDist(i,pmean,Qmean,qreso,qdaq,xtrans)}'
    '''

def get3Parameters(f,npar=3): 
    par = []
    parerr = []
    for i in range(npar): 
        try: 
            f.GetParameter(i)
        except: 
            par.append(1e-10)
            continue
        par.append(f.GetParameter(i))
        parerr.append(f.GetParError(i))
    return par, parerr

def getDataCol(colname, hdfnode): 
    try: 
        coldata = hdfnode.col(colname)
    except: 
        coldata = np.array([])
    return coldata

def parser():
    argparser = ArgumentParser()
    argparser.add_argument('filename', help='Input file name.')
    argparser.add_argument('-t', '--threshold', 
                           type=float, default=20, 
                           help='Threshold for saving waveform')
    argparser.add_argument('-mi','--minimum', 
                           type=int, default=90, 
                           help='Minimum value of the integration window (sampling bin number).')
    argparser.add_argument('-win', '--window', 
                           type=int, default=20,
                           help='Window size. ')
    argparser.add_argument('-bs', '--baselineEst', 
                           type=int, default=160, 
                           help='Starting point of the window to evaluate the baseline value. ')
    argparser.add_argument('--noeachwf',action='store_false')
    argparser.add_argument('--Rev3',action='store_true')
    argparser.add_argument('-k',type=float,default=None)
    argparser.add_argument('--rebin',type=int,default=1)
    argparser.add_argument('--ofile',type=str,default="")
    argparser.add_argument('--xmax',type=float,default=1400)
    argparser.add_argument('--mpe',type=int,default=4)
    argparser.add_argument('--laser',type=int,default=None)
    argparser.add_argument('--poisfix',action='store_true')
    argparser.add_argument('--compwf',action='store_true')

    return argparser.parse_args()


if __name__ == '__main__':
    gROOT.SetStyle("ATLAS")
    main()

