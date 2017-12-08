""" protParser is a Python script originally designed to extract the
    number of exchangable protons in a protein thanks to its sequence
    provided in FASTA format """


import sys
import argParser


#________Main class of the script, modified by decorators
class Protein:

    def __init__(self, seqFile):
        self.seqFile = seqFile
        with open(seqFile, 'r') as seqText:
            self.text = seqText.read().split('\n')


        #_This part goes through listText create two lists with corresponding indexes
        #_ID contains all the ID lines beginning with '>'
        #_seqs contains the sequences corresponding to the ID with the same index        
        self.ID = list()
        self.seqs = list()
        self.seq = ''
        for i, val in enumerate(self.text):
            if '>' in val: 
                self.ID.append(val)
                if self.seq is not '':
                    self.seqs.append(self.seq)
                    self.seq = ''
            elif i == len(self.text)-1:
                self.seqs.append(self.seq)
            else:
                self.seq += val
                


    def __str__(self):
        r = ''
        for i, val in enumerate(self.ID):
            r += "\n\n" + self.ID[i] + "\n"
            r += self.seqs[i] + "\n"
        return r 

#_Decorator of the class Protein
class DecoratorProtein(Protein):

    def __init__(self, protein):
        super().__init__(protein.seqFile)



#________Component of Protein class that uses the sequence
#________to count the number of exchangeable hydrogens
class exHCounter(DecoratorProtein):

    #_Initialisation and creation of a recordtype with amino acids parameters
    def __init__(self, protein):
        super().__init__(protein)
        #_Lists containing infos about th amino acids
        #_0: one-letter code, 1: H weight, 2: exchangeable H, 3: total H, 4: count, 5: methyl
        self.aa = [['A', 89.09, 1, 5, 0, 1],
                  ['C', 121.15, 2, 5, 0, 0],
                  ['D', 133.10, 2, 5, 0, 0],
                  ['E', 147.13, 2, 7, 0, 0],
                  ['F', 165.19, 1, 9, 0, 0],
                  ['G', 75.07, 1, 3, 0, 0],
                  ['H', 155.16, 2, 7, 0, 0],
                  ['I', 131.17, 1, 11, 0, 2],
                  ['K', 146.19, 3, 12, 0, 0],
                  ['L', 131.17, 1, 11, 0, 2],
                  ['M', 149.21, 1, 9, 0, 1],
                  ['N', 132.12, 3, 6, 0, 0],
                  ['P', 115.13, 0, 7, 0, 0],
                  ['Q', 146.15, 3, 8, 0, 0],
                  ['R', 174.20, 5, 12, 0, 0],
                  ['S', 105.09, 2, 5, 0, 0],
                  ['T', 119.12, 2, 7, 0, 1],
                  ['V', 117.15, 1, 9, 0, 2],
                  ['W', 204.23, 2, 10, 0, 0],
                  ['Y', 181.19, 2, 9, 0, 0]]
        

    #_Print the number of exchangeable hydrogens and the mass for H and D forms
    def __str__(self):
        r = ''
        for i, val in enumerate(self.ID):
            massH = 0
            massD = 0
            massHD = 0
            numMeth = 0
            totH = 0
            r += self.ID[i] + "\n"
            r += self.seqs[i] + "\n"
            for j, val in enumerate(self.aa):
                val[4] = self.seqs[i].count(val[0])
                r += "\n Number of {0} : {1} ({2} ex. H)".format(val[0], val[4]
                , val[2] * val[4])
                massH += val[1] * val[4]
                massD += (val[1] + val[3]) * val[4]
                massHD += (val[1] + val[2]) * val[4]
                numMeth += val[4] * val[5]     
                totH += val[4] * val[3] 

            r += '\n\nTotal number of hydrogens: %d' % totH
            r += '\nNumber of methyl groups = %d (%f %s)' % (numMeth, 300*numMeth / totH, '%')
            r += "\n\nTotal hydrogenated mass (Da) = " 
            r += str(int(massH))
            r += "\nTotal deuterated mass (Da) = " 
            r += str(int(massD)) + " (%2.2f" % (100-massH/massD*100) + " %)"
            r += "\nTotal H/D exchanged mass (Da) = " 
            r += str(int(massHD)) + " (%2.2f" % (100-massH/massHD*100) + " %)"
            r += '\n'

            for j, aa in enumerate(self.aa):
                r += '\nFraction of %s: %-3.1f ' % (aa[0], 100 * aa[4] / len(self.seqs[i])) + '%'

            r +='\n\n'

        return r

if __name__ == '__main__':

    prot = exHCounter(Protein(sys.argv[1]))
    for i, value in enumerate(str(prot).split('\n')):
        print(value)

