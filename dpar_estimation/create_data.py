import sys
sys.path.append('./Library')
import lib as lib
import argparse

parser = argparse.ArgumentParser(description='This function creates training data for the ANN')
parser.add_argument('dp_min', type = float, help = 'minimum parallel difusivity i.e 0.5e-3')
parser.add_argument('dp_max', type = float, help = 'maximum parallel difusivity i.e 1e-3')
parser.add_argument('dfree', type = float, help = 'free water difusivity i.e 1.44e-3')
parser.add_argument('bvalfilename', nargs='?', help = 'path and name of the bvals')
parser.add_argument('bvecfilename', nargs='?', help = 'path and name of the bvecs')
parser.add_argument('f_out', nargs='?', help = 'path where the output is going to be stored')
parser.add_argument('shells',nargs="+", type=int, help = 'type the protocol shells ie. -shells 0 500 2000 4500 6000 8000')
parser.add_argument('snr', type=int, default = 22, help = 'type the snr i.e: 22')

args = parser.parse_args()

lib.data_phantom(args.dp_min, args.dp_max, args.dfree, args.bvalfilename, args.bvecfilename, args.shells, args.snr, args.f_out)
