from ROOT import *
import numpy as np
from scipy.stats import poisson, norm
from scipy.special import comb
from argparse import ArgumentParser
import os

def main(): 
    args = parser()
    os.system('mkdir -p plots/model')

    pmean = args.pmean # poisson mean
    qmean = 1 # gain mean 
    qreso = 0.33 # charge resolution 
    qdaq = 0.04 # DAQ resolution 

    mpe = int(pmean * 2) # number of p.e. 
    if args.mpe: 
        mpe = args.mpe
    if mpe < 3: 
        mpe = 5

    maxrange = args.pmean * 100.

    idealfunc = ""
    for i in range(mpe+1): 
        if i!=0:
            idealfunc += "+"
        idealfunc += NphotonDist(i,pmean,i*qmean,i*qreso,qdaq)

    f1 = TF1("idealdist",idealfunc,-1.,maxrange)
    f1.SetNpx(1000)

    gain1 = 1.5 # gain @ 1st dynode 
    gain2 = 2./3. # gain @ 2nd dynode
    loss1 = 0.05 # 5% loss @ 1st dynode

    TwoDynodeModel = ""
    for i in range(mpe+1):
        if i!=0: 
            TwoDynodeModel += "+"
        TwoDynodeModel += f'(1-{loss1})*{NphotonDist(i,pmean,i*gain1*gain2,i*qreso,qdaq)}+{loss1}*{NphotonDist(i,pmean,i*gain2,i*qreso,qdaq)}'

    f2 = TF1("TwoDynodeModel",TwoDynodeModel,-1.,maxrange)
    f2.SetNpx(1000)

    a = np.array([16.8, 4, 5, 3.33, 1.67, 1, 1.2, 1.5, 2.2, 3, 2.4]) # R5912-100 Divider
    Total = a.sum() 
    R = a/Total # Divider Ratio 

    # gain is calculated by g = A^{n} * R_1 * ...* R_n * V^{kn} 
    # where A, k are constant, R_n is the divider ratio, and V is the bias voltage. 

    k = 0.75
    R2_10 = R[1:10].prod()
    A = 1./ R[0:10].prod()**(k/10.)
    gain1 = A * R[0]**k
    gain2_10 = A**9 * R2_10**k
    print (f'{R2_10}, {A},  {gain1}, {gain2_10}, {gain1*gain2_10}')
    
    R5912Simple = ""
    for i in range(mpe+1):
        if i!=0: 
            R5912Simple += "+"
        R5912Simple += f'(1-{loss1})*{NphotonDist(i,pmean,i*gain1*gain2_10,i*qreso,qdaq)}+{loss1}*{NphotonDist(i,pmean,i*gain2_10,i*qreso,qdaq)}'


    f3 = TF1("R5912Simple",R5912Simple,-1.,maxrange)
    f3.SetNpx(1000)

    R5912Model = "0"
    for i in range(mpe+1): 
        for j in range(i+1): 
            R5912Model += f'+{comb(i,j)*((1-loss1)**j)*(loss1**(i-j))}*{NphotonDist(i,pmean,j*gain1*gain2_10+(i-j)*gain2_10,qreso,qdaq)}'

    print(R5912Model)

    f4 = TF1("R5912Model",R5912Model,-1.,maxrange)
    f4.SetNpx(1000)

    g1 = gain1 
    g2 = A * R[1]**k
    g3_10 = A**8 * R[2:10].prod()**k
    R5912ModelN2 = "0" 
    for i in range(mpe+1): 
        for j in range(i+1): 
            for l in range(i-j+1): 
                R5912ModelN2 += f'+{comb(i,j)*((1-loss1)**j)*(loss1**(i-j))*comb(i-j,l)*((1-loss1)**l)*(loss1**(i-j-l))}*{NphotonDist(i,pmean,j*g1*g2*g3_10+l*g2*g3_10+(i-j-l)*g3_10,qreso,qdaq)}'

    f5 = TF1("R5912_N2",R5912ModelN2,-1.,maxrange)
    f5.SetNpx(1000)

    loss2 = 0.05
    R5912ModelN2_2 = "0"
    for i in range(mpe+1):
        for j in range(i+1): 
            for l in range(i-j+1): 
                R5912ModelN2_2 += f'+{(1-loss2)*comb(i,j)*((1-loss1)**j)*(loss1**(i-j))*comb(i-j,l)*((1-loss2)**l)*(loss2**(i-j-l))}*{NphotonDist(i,pmean,j*g1*(1-loss2)*g2*g3_10+l*g2*g3_10+(i-j-l)*g3_10,qreso,qdaq)}+{loss2*comb(i,j)*((1-loss1)**j)*(loss1**(i-j))*comb(i-j,l)*((1-loss2)**l)*(loss2**(i-j-l))}*{NphotonDist(i,pmean,j*g1*loss2*g3_10+l*g2*g3_10+(i-j-l)*g3_10,qreso,qdaq)}'

    f6 = TF1("R5912_N2_2",R5912ModelN2_2,-1.,maxrange)
    f6.SetNpx(1000)

    R5912ModelN2_3 = "0"
    for i in range(mpe+1):
        for j in range(i+1): 
            R5912ModelN2_3 += f'+{(1-loss2)*comb(i,j)*((1-loss1)**j)*(loss1**(i-j))}*{NphotonDist(i,pmean,j*g1*(1-loss2)*g2*g3_10+(i-j)*g2*g3_10,qreso,qdaq)}+{loss2*comb(i,j)*((1-loss1)**j)*(loss1**(i-j))}*{NphotonDist(i,pmean,j*g1*loss2*g3_10+(i-j)*g2*g3_10,qreso,qdaq)}'

    f7 = TF1("R5912_N2_3",R5912ModelN2_3,-1.,maxrange)
    f7.SetNpx(1000)

    subtf45 = TF1("subtf45","R5912Model-R5912_N2",-1.,maxrange)
    subtf45.SetNpx(1000)

    Pedpeak = f1.Eval(0)
    SPEpeak = f1.Eval(1)
    
    f = TFile("rootfiles/testplot.root","recreate")
    f1.Write()
    f2.Write()
    f3.Write()
    f4.Write()
    f5.Write()
    f6.Write()
    subtf45.Write()
    f.Close()

    xmax = 3.5
    if args.pmean > 2: 
        xmax = args.pmean*10
    if args.xmax: 
        xmax = args.xmax
    ymax = Pedpeak*1.1
    if ymax < f1.GetMaximum()*1.1: 
        ymax = f1.GetMaximum()*1.1
    if ymax < f4.GetMaximum()*1.1:
        ymax = f4.GetMaximum()*1.1
    if args.ymax: 
        ymax = args.ymax
    c = TCanvas("c","c",800,600)
    c.Draw()
    h = TH1D("",";#Photo-Electrons;Probability",11000,-100,1000)
    h.GetXaxis().SetRangeUser(-0.5,xmax)
    #h.GetYaxis().SetRangeUser(0,SPEpeak*1.2)
    h.GetYaxis().SetRangeUser(0,ymax)
    f1.SetLineColor(1)
    f2.SetLineColor(6)
    f3.SetLineColor(4)
    f4.SetLineColor(2)
    f5.SetLineColor(9)
    f6.SetLineColor(46)
    f7.SetLineColor(38)
    h.Draw()
    f1.Draw("Lsame")
    c.SaveAs(f"plots/model/ideal_p{args.pmean}.pdf")
    h.Draw()
    f4.Draw("Lsame")
    c.SaveAs(f"plots/model/model_p{args.pmean}.pdf")
    h.Draw()
    f1.Draw("Lsame")
    h.GetYaxis().SetRangeUser(0,SPEpeak*1.1)
    c.SaveAs(f"plots/model/expand_{args.pmean}.pdf")
    h.GetXaxis().SetRangeUser(-0.5,4)
    #f2.Draw("Lsame")
    #f3.Draw("Lsame")
    f4.Draw("Lsame")
    #f5.Draw("Lsame")
    #f6.Draw("Lsame")
    #f7.Draw("Lsame")

    leg = TLegend(.7,.7,.9,.9)
    leg.AddEntry(f1,"Ideal","L")
    leg.AddEntry(f4,"R5912Model","L")
    leg.Draw()

    c.SaveAs(f"plots/model/cmp_{args.pmean}.pdf")


    return

def NphotonDist(Num, PoisMean, QMean, QReso, Qdaq): 
    return  f'TMath::PoissonI({Num},{PoisMean}) * TMath::Gaus(x, {QMean}, {np.sqrt(QMean*QReso**2+Qdaq**2)},1)'

def NgammaDist(Num, PoisMean, QMean, QReso, Qdaq, xtrans=0): 
    return  f'TMath::PoissonI({Num},{PoisMean}) * TMath::Gaus(x-{xtrans}, [1]*{QMean}, TMath::Sqrt([1]*[1]*{QMean*QReso**2}+{Qdaq**2}),1)'

def parser():
    argparser = ArgumentParser()
    argparser.add_argument('--mpe',type=int,default=None)
    argparser.add_argument('--pmean',type=float,default=0.5)
    argparser.add_argument('--xmax',type=float,default=None)
    argparser.add_argument('--ymax',type=float,default=None)
    return argparser.parse_args()

if __name__ == "__main__": 
    gROOT.SetStyle("ATLAS")
    main()
