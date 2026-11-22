# -*- coding: utf-8 -*-
"""Export the raw data behind all figures to CSV files, so the plots can be
rebuilt independently (e.g. in Origin).

Reads JSON data from <repo>/data/ and writes CSVs into <repo>/data/figure_data_csv/.
"""
import json, os, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
CSVDIR = os.path.join(DATA, 'figure_data_csv')
os.makedirs(CSVDIR, exist_ok=True)

def w(name, header, rows):
    with open(os.path.join(CSVDIR, name), 'w', newline='', encoding='utf-8-sig') as f:
        wr = csv.writer(f); wr.writerow(header); wr.writerows(rows)
    print('written', name)

with open(os.path.join(DATA, 'unified_p3_final.json'), encoding='utf-8') as f:
    uni = json.load(f)
DOMAINS = ['0.014@0','0.014@1','0.014@2','0.014@3',
           '0.021@0','0.021@1','0.021@2','0.021@3']
REPS = ('plain','env','order')

# fig2
snr = ['clean','10','0','-5']
w('fig2_snr.csv', ['test_snr_db','plain_clean','env_clean','plain_noise','env_noise'],
  list(zip(snr, [100,99,65,26.5],[100,100,89,30.5],[100,100,100,100],[98,100,100,97.5])))

# fig3 heatmap + dumbbell
mat = {r: [uni['%s|42' % r][d]['acc'] for d in DOMAINS] for r in REPS}
w('fig3a_accuracy_matrix.csv', ['domain','plain','env','order'],
  [(d, mat['plain'][i], mat['env'][i], mat['order'][i]) for i, d in enumerate(DOMAINS)])
ms_p = os.path.join(DATA, 'multiseed_recall.json')
if os.path.exists(ms_p):
    with open(ms_p, encoding='utf-8') as f: ms = json.load(f)
    w('fig3b_plain_seeds.csv', ['domain','seed1','seed2','seed3','mean','std'],
      [(d,) + tuple(ms[d]['acc']) + (float(np.mean(ms[d]['acc'])), float(np.std(ms[d]['acc'])))
       for d in DOMAINS if d in ms])

# fig4 confusion (display order N,IR,OR,B)
order = [2,0,1,3]
for dom, tag in [('0.014@0','fig4a_conf_0014_0HP.csv'), ('0.021@0','fig4b_conf_0021_0HP.csv')]:
    c = np.array(uni['plain|42'][dom]['conf'])[np.ix_(order, order)]
    w(tag, ['true\\pred','N','IR','OR','B'],
      [(['N','IR','OR','B'][i],) + tuple(int(v) for v in c[i]) for i in range(4)])

# fig5 fisher
with open(os.path.join(DATA, 'fisher_12domains.json'), encoding='utf-8') as f:
    fd = json.load(f)
doms = [(s,l) for s in ('0.007','0.014','0.021') for l in range(4)]
bands = ['0-0.5','0.5-1','1-1.5','1.5-2','2-2.5','2.5-3','3-3.5','3.5-4']
mat5 = np.array([fd['%s@%d' % (s,l)] for s,l in doms])
w('fig5a_fisher_matrix.csv', ['domain'] + bands,
  [('%s@%dHP' % doms[i],) + tuple(float(v) for v in mat5[i]) for i in range(12)])
w('fig5b_peak_band.csv', ['load_HP','0.007in','0.014in','0.021in'],
  [(l, bands[mat5[0*4+l].argmax()], bands[mat5[1*4+l].argmax()], bands[mat5[2*4+l].argmax()])
   for l in range(4)])

# fig6
w('fig6_remedies.csv', ['remedy','delta_points'],
  [('Fixed 4-4.8 kHz band',4.63),('Band-masking training',1.87),
   ('Multi-source augmentation',0.25),('Entropy TTA',-3.13),
   ('Envelope demodulation',-5.50),('Order normalisation',-5.56),
   ('Entropy band selection',-9.56),('BN adaptation',-15.87)])

# fig7
w('fig7a_datasets.csv', ['dataset','in_domain','transfer'],
  [('CWRU cross-severity',100.00,62.25),('JNU cross-speed',99.89,73.83),
   ('Paderborn cross-size',48.22,53.56)])
rec = [[17.33,100,43.33,96],[3.33,100,48.67,96],[54.67,100,54,98.67],
       [50.67,100,100,100],[0,99.33,20,100],[91.33,98.67,100,100]]
w('fig7b_jnu_recall.csv', ['task','Normal','Inner race','Outer race','Ball'],
  [('600->800',)+tuple(rec[0]),('600->1000',)+tuple(rec[1]),('800->600',)+tuple(rec[2]),
   ('800->1000',)+tuple(rec[3]),('1000->600',)+tuple(rec[4]),('1000->800',)+tuple(rec[5])])
print('done ->', CSVDIR)
