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
    os.system('mkdir -p plots')

    f = tables.open_file(filename)

    data = f.get_node('/data')
    event_ids = data.col('event_id')
    times = data.col('time')
    waveforms = data.col('waveform')

    try: 
        timestamps = data.col('timestamp') 
    except: 
        print("there is no timestamp in this file.")
        timestamps = np.arange(len(waveforms))

    nsamples = len(waveforms[0])
    fq = np.linspace(0,240,nsamples)

    gmean = np.mean(waveforms[0])
    globalmean = np.mean(waveforms)
    print(np.mean(waveforms[0]))

    ofilename = filename.split('/')[len(filename.split('/'))-1].split('.h')[0] + args.ofile

    of = TFile(f'rootfiles/{ofilename}.root',"RECREATE")

    h     = TH1D('qdist','qdist;ADC count [LSB];Entry',500,-100,900)
    hopt  = TH1D('optq','optq;ADC count [LSB];Entry', 1000, -100,1900)
    hoptq = TH1D('optqcal','optqcal;Charge [pC];Entry', 500, -1,9)
    hpk   = TH1D('peak','peak;ADC count [LSB];Entry',500,-100,900)
    havg  = TH1D('avgwf','Averaged Waveform;Sampling Bin;ADC count',nsamples,0,nsamples)
    
    winmin = args.minimum
    winmax = args.minimum + args.window
    bsstart = args.baselineEst

    print (f'Total #Events: {len(waveforms)}')
    print (f'Setting... Window [{winmin}:{winmax}] and Pedestal start from {bsstart}.\n')

    topdir = gDirectory.GetDirectory(gDirectory.GetPath())
    subdir = topdir.mkdir("Waveforms")

    fltwfs = []
    starttimestamp = timestamps[0]

    Mbcalib = 0.075e-3 # V/LSB for Rev4
    if args.Rev3: 
        Mbcalib = 0.0787e-3 #V/LSB
    Mbspfq = 240.e6 # Hz sampling freq
    Mbimped = 50. # ohm impedance
    elecQ = 1.60217662e-19 # elementary charge

    for i in tqdm(range(len(waveforms))):
        waveform = waveforms[i]
        timestamp = timestamps[i]

        baseline_mean = np.mean(waveform)
        selected = waveform[waveform < np.mean(waveform) + 10]
        reduced_waveform = waveform - baseline_mean
        scale = (nsamples-bsstart)/(winmax-winmin)
        h.Fill(sum(reduced_waveform[winmin:winmax]))

        maxbin = np.argmax(reduced_waveform)
        if maxbin < 50: 
            maxbin = 50
        if maxbin > nsamples - 20: 
            maxbin = nsamples - 20
        hopt.Fill(sum(reduced_waveform[maxbin-10:maxbin+10]))
        hoptq.Fill(sum(reduced_waveform[maxbin-10:maxbin+10])*Mbcalib/Mbimped/Mbspfq*1.e12)

        hpk.Fill(max(waveform[winmin:winmax])-np.mean(waveform[bsstart:nsamples]))

        if max(waveform) - np.mean(waveform[bsstart:nsamples]) < args.threshold:
            continue

        fltwfs.append(waveform)

        if args.noeachwf: 
            h2 = TH1D(f'w{i}','Waveform{i};Sampling Bin;ADC count',nsamples,0,nsamples)
            for j in range(len(waveform)):
                h2.Fill(j,waveform[j])
            subdir.cd()
            h2.Write()

    print('')

    avgfltwfs = np.mean(fltwfs, axis=0)
    for i in range(len(avgfltwfs)): 
        havg.Fill(i,avgfltwfs[i])

    topdir.cd()
    h.Write()
    hopt.Write()
    hoptq.Write()
    hpk.Write()
    havg.Write()

    rebin = args.rebin
    hopt.Rebin(rebin)
    hsubt = hopt.Clone()
    hopt.Fit('gausn',"","",-20,20)
        
    fped = hopt.GetFunction('gausn')
    xtrans = hopt.GetFunction('gausn').GetParameter(1)
    pmean = -1. * np.log(hopt.GetFunction('gausn').GetParameter(0)/2./rebin/hopt.GetEntries()) #0.3 # poisson mean
    qreso = 0.3 # charge resolution 
    qdaq = hopt.GetFunction('gausn').GetParameter(2) #0.07 # DAQ resolution 
    loss1 = 0.07 # 7% loss @ 1st dynode

    fitmax = 380
    hopt.GetXaxis().SetRangeUser(xtrans+qdaq*5,1900) 
    maxfitrange = hopt.GetBinCenter(hopt.GetMaximumBin())
    hopt.GetXaxis().UnZoom()
    interval = maxfitrange - (xtrans+qdaq*5)
    if interval < 1: 
        interval = 5
    print(f'Est SPE Mean: {maxfitrange}, Interval: {interval}')

    hopt.Fit('gaus',"","",xtrans+qdaq*5,maxfitrange+interval)
    est1mean = hopt.GetFunction('gaus').GetParameter(1)
    lowlimit = qdaq*5 
    if lowlimit < est1mean/2.: 
        lowlimit = est1mean/2.
    hopt.Fit('gaus',"","",lowlimit,est1mean*1.5)
    fspe = hopt.GetFunction('gaus')
    qreso = np.sqrt(hopt.GetFunction('gaus').GetParameter(2)**2-qdaq**2)/(hopt.GetFunction('gaus').GetParameter(1)-xtrans)

    k = 0.75
    mpe = 3 # number of p.e. 
    a = [16.8, 4, 5, 3.33, 1.67, 1, 1.2, 1.5, 2.2, 3, 2.4] # R5912-100 Divider list 
    anpy = np.array(a) 
    
    Total = np.sum(anpy) #16.8 + 4 + 5 + 3.33 + 1.67 + 1 + 1.2 + 1.5 + 2.2 + 3 + 2.4 # R5912-100 Divider
    Rnpy = anpy/Total # Divider Ratio 

    R2_10 = np.prod(Rnpy[1:10])#4 * 5 * 3.33 * 1.67 * 1 * 1.2 * 1.5 * 2.2 * 3 / Total**9
    A = 1./ (np.prod(Rnpy[0:10])**(k/10.))
    gain1 = A * Rnpy[0]**k
    rgain1  = Rnpy[0]/np.prod(Rnpy[0:10]**0.1)
    rgain2_ = R2_10/(np.prod(Rnpy[0:10])**0.9)
    rgain3_ = np.prod(Rnpy[2:10])/(np.prod(Rnpy[0:10])**0.8)

    R5912Model = "0"
    for i in range(mpe+1): 
        for j in range(i+1): 
            # Model B 
            Qmean = f'({j}+{(i-j)}*pow({rgain2_},[3]))'
            R5912Model += f'+[0]*{comb(i,j)}*pow((1-[2]),{j})*pow([2],{i-j})*{NphotonDist(i,pmean,Qmean,qreso,qdaq,xtrans)}'

    f4 = TF1("R5912Model",R5912Model,-100.,1900.)
    f4.SetLineColor(2)
    f4.SetNpx(500)

    preQMean = hopt.GetFunction('gaus').GetParameter(1)

    f4.SetParameter(1,preQMean)
    f4.SetParameter(2,loss1)
    f4.SetParLimits(2,0,0.5)
    if args.k: 
        f4.SetParameter(3,float(args.k))
        f4.SetParLimits(3,0.75,0.75)
    else: 
        f4.SetParameter(3,0.75)
        f4.SetParLimits(3,0.6,0.9)
    hopt.Fit("R5912Model","","",-100,1900)

    hsubt.SetName("hsubt")
    hsubt.SetTitle("hsubt;ADC count [LSB];Data - Fit")
    hsubt.Add(f4,-1)

    print(f'{pmean}, {xtrans}, {qdaq}, {f4.GetParameter(0)}')
    print(f'Observed gain: {f4.GetParameter(1)*Mbcalib/Mbimped/Mbspfq/elecQ:.5e}')

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
    hPars.SetBinError(5, 1./(fped.GetParameter(0))*fped.GetParError(0))
    hPars.GetXaxis().SetBinLabel(6,"Pedestal Mean [LSB]")
    hPars.SetBinContent(6,xtrans)
    hPars.SetBinError(6, fped.GetParError(1))
    hPars.GetXaxis().SetBinLabel(7,"Pedestal Sigma [LSB]")
    hPars.SetBinContent(7,qdaq)
    hPars.SetBinError(7, fped.GetParError(2))
    hPars.GetXaxis().SetBinLabel(8,"Total Gain [10^{7}]")
    hPars.SetBinContent(8,f4.GetParameter(1)*Mbcalib/Mbimped/Mbspfq/elecQ/1.e7)
    hPars.SetBinError(8,f4.GetParError(1)*Mbcalib/Mbimped/Mbspfq/elecQ/1.e7)
    hPars.GetXaxis().SetBinLabel(9,"Charge Resolution [LSB]")
    hPars.SetBinContent(9,qreso)
    hPars.SetBinError(9,np.sqrt(fspe.GetParError(2)**2-fped.GetParError(2)**2)/(fspe.GetParameter(1)-xtrans))
    hPars.GetXaxis().SetBinLabel(10,"Peak to Valley Ratio")
    hPars.SetBinContent(10,peak/valley)
    hPars.GetXaxis().SetBinLabel(11,"1st Dynode Loss Probability")
    hPars.SetBinContent(11,f4.GetParameter(2))
    hPars.SetBinError(11,f4.GetParError(2))
    hPars.GetXaxis().SetBinLabel(12,"Slope Constant of Gain Curve") 
    hPars.SetBinContent(12,f4.GetParameter(3))
    hPars.SetBinError(12,f4.GetParError(3))

    hPars.Write()

    gStyle.SetPadGridX(1)
    gStyle.SetPadGridY(1)
    c = TCanvas("c","c",1000,800)
    c.Draw()
    hopt.SetMarkerStyle(20)
    hopt.SetMarkerSize(1)
    rp = TRatioPlot(hopt)
    rp.SetH1DrawOpt("PE")
    rp.SetFitDrawOpt("L")
    rp.SetGraphDrawOpt("PE")
    rp.Draw()
    rp.GetLowerRefYaxis().SetTitle("Data/Fit")
    rp.GetUpperRefYaxis().SetTitle("Entries")
    rp.GetUpperRefYaxis().SetRangeUser(0,peak*1.5)
    rp.GetUpperRefXaxis().SetRangeUser(-50,1900)
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

    return argparser.parse_args()


if __name__ == '__main__':
    gROOT.SetStyle("ATLAS")

    main()

