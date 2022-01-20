import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--begin', metavar='1', type=int, nargs=1,
                    help='begin loop', default=1)
parser.add_argument('--end', metavar='1', type=int, nargs=1,
                    help='end loop', default=1)
parser.add_argument('--mask', metavar='1', type=str, nargs=1,
                    help='end loop', default=1)

args = parser.parse_args()
begin = args.begin[0]
end = args.end[0]
mask = args.mask[0]

path_to_sim = '/data/gravwav/lopezm/MasterThesis/Results/SimulatedMaps/'


for types in ['Pearson coefficient']:
    
    if types == 'Norm':
        name = 'Norm'
    else:
        name = 'Pearson'

    path_to_store = path_to_sim+str(name)+'/'+str(mask)
    for s in range(begin, end):
        data_sim = pd.DataFrame()
        rows = dict()
        for i in range(0,1000):

            df = pd.read_pickle(path_to_sim+'/'+str(mask)+'/sim_map_'+str(i)+'_'+str(mask)+'.csv')
            df = df.reset_index(drop=True)
            rows[str(types)+' map '+str(i)] = df[types][s]
            #data_sim[str(types)+' map '+str(i)] = df[types][s]
        data_sim = pd.DataFrame(rows)
        data_sim.to_pickle(path_to_store+'/'+str(name)+'_'+str(s)+'_'+str(mask)+'.csv')
