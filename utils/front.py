import datetime
from typing import Union
from tqdm import tqdm

from .back import ArtifactManager

class LogInterface:
    def __init__(self, am: ArtifactManager):
        self.am = am
        self.tbar = None

        self.info(f"Conductor --- {datetime.datetime.now()}")  # init console logger

    def info(self, content: Union[str, list], fn: bool = False, bn: bool = False):
        if isinstance(content, list) and len(content) == 0:
            return

        if fn:
            self.info('')
            
        if isinstance(content, str):
            print(content)
            self.am.info(content)
        elif isinstance(content, list):                
            for row in content:
                self.info(row)

        if bn:
            self.info('')

    def metrics(self, met: dict):
        k_str, v_str = '', ''
        for k, v in met.items():
            if isinstance(v, int):
                v = f"{v:<15}"
            elif isinstance(v, float):
                if k == 'time':
                    v = f"{v:<15.2f}"
                else:
                    v = f"{v:<15.6f}"
            k_str += f"{k:<15}"
            v_str += v
        self.info([k_str, v_str])
        self.am.metrics(met)

    def bar_init(self, total: int, desc: str):
        self.info('')
        self.tbar = tqdm(total=total, desc=desc, bar_format="{l_bar}{bar:40}{r_bar}")

    def bar_update(self, desc: str = None):
        assert self.tbar, f"expect init_bar before update_bar"
        self.tbar.update()
        if desc:
            self.tbar.set_description(desc=desc)

    def bar_close(self):
        assert self.tbar, f"expect init_bar before close_bar"
        self.am.info(f"[process bar] ({self.tbar.total} - done) {self.tbar.desc}")
        self.tbar.close()
        self.tbar = None
