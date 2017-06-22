import argparse
import numpy as np
def do_operation(args):
	
	if(args.operation=='add'):
		return np.sum(np.array(args.values))
	elif(args.operation=='mul'):
		return np.product(np.array(args.values))
	else:
		return 0
	
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--operation",help="Operation to be carried out. Possible operations are add,mul")
	parser.add_argument('--values',nargs='*',help="Space seperated values on which the operation has to be done",type=float)
	
	
	args = parser.parse_args()
	print(do_operation(args))
if __name__=='__main__':
	main()