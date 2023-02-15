import re


class c_fct:
    r_type = None
    name   = ""
    args_types = []
    args_names = []

    def __init__(self, line):
        first_space = line.find(' ')
        (self.r_type, name_args)  = (line[0:first_space], line[first_space+1:-1])

        first_par = name_args.find('(')
        (self.name, args_list) = (name_args[6: first_par], name_args[first_par+1:-2].split(', '))
        
        self.args_names = [arg_i[arg_i.rfind(" ")+1:].replace('*','').lower() for arg_i in args_list]
        print(f'{self.name}\t{self.args_names}')

def fcts_in(filename):
    with open(filename, 'r') as f:
        f_fcts = [c_fct(line) for line in f.readlines()]
        return f_fcts
