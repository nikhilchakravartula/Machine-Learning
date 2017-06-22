import argparse

def do_operation(args):
	print(args)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--operation",help="Addition")
	
	args = parser.parse_args()
	do_operation(args)
if __name__=='__main__':
	main()