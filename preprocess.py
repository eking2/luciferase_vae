import requests
from requests.exceptions import HTTPError
import subprocess
import time
from pathlib import Path


class PrepLucDataset:

    def __init__(self, interpro='IPR011251', pfam='PF00296', seq_len='504', seq_thresh=0.7, valid_ratio=0.2):

        self.interpro = interpro
        self.pfam = pfam
        self.seq_len = seq_len
        self.seq_thresh = seq_thresh
        self.valid_ratio = valid_ratio

        self.seqs = None

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

        pass


    def run_mmseqs2(self):

        '''cluster sequences on seq_threshold'''

        pass


    def split_data(self):

        '''randomly split clusters into train/valid sets'''

        pass


    def download_pfam_hmm(self):

        '''download pfam hmm to use as clustal omega external profile alignment'''

        save_path = Path(f'data/{self.pfam}.hmm')

        if not save_path.exists():
            print(f'Downloading {self.pfam}.hmm')
            url = f'https://pfam.xfam.org/family/{self.pfam}/hmm'
            r = requests.get(url)
            r.raise_for_status()

            save_path.write_text(r.text)


if __name__ == '__main__':

    prep = PrepLucDataset()
    prep.download_interpro()
    #prep.download_pfam_hmm()
