'''
Created on 2010-07-26

@author: Shobhit Jain

@contact: shobhit@cs.toronto.edu

----------------------------------------------------------------------

Modified on 2023-04-13 by Anthony Van Cayetano
Changes:
    - This modified file is the Python 3 version of the original
        file (which was originally written in Python 2). The conversion
        was done via the builtin Python script called '2to3'. 
        The said script also created a backup of the original file,
        with the name 'parser.py.bak'.

    - The '_go_annotations' function was modified by Anthony Van Cayetano. 
        In particular, the line 'gene = line[1].strip()' inside the said 
        function was changed to 'gene = line[10].strip().split("|")[0]'.

'''

class Parser(object):
    '''
    Parser class implement functions for parsing input data files.
    '''
    
    def __init__(self):
        '''
        Constructor
        '''
        super(Parser, self).__init__()
        
        
     
    def _obo_parser(self, obo_file):    
        '''
        Parser for Gene Ontology obo files. go_annotations variable is 
        updated.
        '''
        file = open(obo_file, 'r')
        flag = 0
        for line in file:
            line = line.strip()
            if line.startswith('[Term]'): 
                flag = 1
                node = ''
                name = ''
                domain = ''
                parent = set()
            elif flag == 1 and line == '': 
                flag = 0
                self._add_node(node)
                for term in parent:
                    self._add_node(term)
                    self._add_edge(term, node)
                self.go_annotations[node] = {'name':name, 
                                             'domain':domain, 
                                             'gene':set(), 
                                             'ancestors':set(), 
                                             'cluster':{}
                                             }
            elif flag == 1 and line.startswith("id"):
                node = line.split('id:')[1].strip()   
            elif flag == 1 and line.startswith("namespace"):
                domain = line.split('namespace:')[1].strip()
            elif flag == 1 and line.startswith("name"):
                name = line.split('name:')[1].strip()
            elif flag == 1 and line.startswith("is_a"):
                parent.add(line.split(' ')[1])
            elif flag == 1 and line.startswith("relationship"):
                parent.add(line.split(' ')[2])
                
                
                
    def _go_annotations(self, gene_file, cd):
        '''
        Parser for gene annotation file (SGD/human). go_annotations variable is updated.
        '''
        file = open(gene_file, 'r')
        for line in file:
            line = line.strip()
            if line != "" and not line.startswith('!'):
                line = line.split('\t')
                term = line[4].strip()
                gene = line[10].strip().split("|")[0] # This line was modified by Anthony Van Cayetano on 2023-04-13.
                code = line[6].strip()
                if term in self.go_annotations and code != cd:
                    self.go_annotations[term]['gene'].add(gene)
                    
                
                
                

                        
                        
