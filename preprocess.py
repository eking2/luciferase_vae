import requests
from requests.exceptions import HTTPError
import subprocess
import shlex
import time
from Bio import SeqIO
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

class PrepLucDataset:

    def __init__(self, interpro='IPR011251', pfam='PF00296', seq_len='504', seq_thresh=0.7):

        self.interpro = interpro
        self.pfam = pfam
        self.seq_len = int(seq_len)
        self.seq_thresh = seq_thresh


    def download_interpro(self):

        '''download all luciferase sequences from interpro'''

        BASE_URL = f"https://www.ebi.ac.uk:443/interpro/api/protein/UniProt/entry/InterPro/{self.interpro}/?page_size=200&extra_fields=sequence"
        url = BASE_URL

        # make output fasta
        out_path = Path(f'./data/{self.interpro}.fasta')
        if not out_path.exists():
            out_path.touch()
        else:
            return

        # read each page of results and write to output fasta
        with open(out_path, 'w') as f:
            print(f'Downloading {self.interpro}.fasta')

            while url:
                print(url)
                try:
                    r = requests.get(url, headers={'Accept' : 'application/json'})

                    # pause after timeout
                    # continue from same url
                    if r.status_code == 408:
                        time.sleep(61)
                        continue

                    # no data
                    elif r.status_code == 204:
                        break

                    payload = r.json()
                    url = payload.get('next', None)
                    attempts = 0

                except HTTPError as e:
                    if e.response.status_code == 408:
                        time.sleep(61)
                        continue

                    else:
                        # retry 3 times
                        if attempts < 3:
                            attempts += 1
                            time.sleep(61)
                            continue

                        else:
                            print(f'LAST URL: {url}')
                            raise e

                # write records
                for i, item in enumerate(payload['results']):

                    accession = item['metadata']['accession']
                    name = item['metadata']['name']
                    seq_len = item['metadata']['length']
                    start = item['entries'][0]['entry_protein_locations'][0]['fragments'][0]['start']
                    end = item['entries'][0]['entry_protein_locations'][0]['fragments'][0]['end']

                    seq = item['extra_fields']['sequence']
                    seq_wrap = '\n'.join([seq[0+i : 80+i] for i in range(0, len(seq), 80)])

                    out = f'>{accession} | {name} | {seq_len} | {start}...{end}\n{seq_wrap}\n'

                    f.write(out)

                # pause after each page
                if url:
                    time.sleep(1)


    def clean_interpro(self):

        '''delete all sequences longer than seq_len'''

        records = SeqIO.parse(f'./data/{self.interpro}.fasta', 'fasta')

        out_path = Path(f'./data/clean_{self.interpro}.fasta')

        if not out_path.exists():
            print(f'Filtering out all sequences longer than {self.seq_len} residues from {self.interpro}')
            with open(out_path, 'w') as f:

                filtered = (record for record in records if len(record.seq) <= self.seq_len)
                SeqIO.write(filtered, f, 'fasta')


    def run_mmseqs2(self):

        '''cluster sequences on seq_threshold'''

        print('Running mmseqs2')
        cmd = f'mmseqs easy-cluster ./data/clean_{self.interpro}.fasta ./data/clusterRes ./data/tmp --min-seq-id {self.seq_thresh}'
        subprocess.run(shlex.split(cmd))


    def download_pfam_hmm(self):

        '''download pfam hmm to use as clustal omega external profile alignment'''

        save_path = Path(f'data/{self.pfam}.hmm')

        if not save_path.exists():
            print(f'Downloading {self.pfam}.hmm')
            url = f'https://pfam.xfam.org/family/{self.pfam}/hmm'
            r = requests.get(url)
            r.raise_for_status()

            save_path.write_text(r.text)


def get_ipro(accession):

    '''get interpro accession ids from uniprot'''

    url = f'https://www.uniprot.org/uniprot/{accession}.xml'
    r = requests.get(url)
    r.raise_for_status()

    content = BeautifulSoup(r.text)
    records = content.find_all('dbreference')

    interpro_ids = []
    for record in records:
        if record['type'] == 'InterPro':
            interpro_ids.append(record['id'])

    return interpro_ids


def get_valid_ipros(fasta):

    '''download interpro accession ids for all samples'''

    records = SeqIO.parse(fasta, 'fasta')

    with open('/data/ipros_out.txt', 'w') as f:
        for record in records:

            time.sleep(0.1)

            try:
                interpro_ids = get_ipro(record.id)
                res = f'{record.id} - {",".join(interpro_ids)}\n'

                f.write(res)

            except:
                f.write(f'skip {record.id}')


def tidy_interpro(interpros):

    '''convert raw text of interpros to tidy format'''

    lines = Path(interpros).read_text().splitlines()

    df = pd.DataFrame(columns=['accession', 'ipro'])

    for line in tqdm(lines):
        if '-' in line:
            accession = line.split('-')[0].strip()
            ipros = line.split('-')[1].strip().split(',')

            for ipro in ipros:
                temp_df = pd.DataFrame({'accession' : accession,
                                        'ipro' : [ipro]})

                df = df.append(temp_df, ignore_index=True)

    df.to_csv('data/valid_ipro_labels.csv', index=False)
    # df_filtered = df.query("ipro in @save")
    # df_filtered.loc[:, 'ipro'] = df_filtered['ipro'].astype('category')
    # df_filtered.loc[:, 'ipro'] = df_filtered['ipro'].cat.reorder_categories(save)
    # df_filtered.sort_values(by=['ipro', 'accession'], inplace=True)
    # df_filtered.drop_duplicates('accession', inplace=True)
    # df_filtered.to_csv('./data/df_filtered.csv', index=False)


if __name__ == '__main__':

    #prep = PrepLucDataset()
    #prep.download_pfam_hmm()
    #prep.download_interpro()
    #prep.clean_interpro()
    #prep.run_mmseqs2()

    #out = get_ipro('A1JMY1')
    #get_valid_ipros('./data/luxafilt_llmsa_val.fa')

    #save = ['IPR016215', 'IPR019949', 'IPR019952', 'IPR019945', 'IPR022290',
    #        'IPR019911', 'IPR019951', 'IPR023934', 'IPR024014']
    tidy_interpro('./data/ipros_out.txt')
