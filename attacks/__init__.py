# attacks/__init__.py
from .attack import Attack

# 逐一显式导入每种攻击的类（类名你可以沿用下面的命名；若不同就在这里改注册表即可）
from .pgd import PGD_Linf          # class PGD_Linf(Attack)
from .pgdl2 import PGD_L2          # class PGD_L2(Attack)
from .cw import CW                 # class CW(Attack)
from .knn import KNN               # class KNN(Attack)
from .drop import PointDrop        # class PointDrop(Attack)
from .add import PointAdd          # class PointAdd(Attack)
from .vanila import VANILA         # class VANILA(Attack)

# 统一的攻击注册表：键是命令行传进来的 --attack 字符串，值是类对象
ATTACK_REGISTRY = {
    # 你可以把等价名都映射进来，大小写不敏感我在 eval 里会lower()
    'pgd_linf': PGD_Linf,
    'pgdlinf': PGD_Linf,
    'pgd': PGD_Linf,               # 习惯用法
    'pgd_l2': PGD_L2,
    'pgdl2': PGD_L2,

    'cw': CW,
    'knn': KNN,

    'pointdrop': PointDrop,
    'drop': PointDrop,

    'pointadd': PointAdd,
    'add': PointAdd,

    'vanila': VANILA,              # 你文件叫 vanila.py（注意不是 vanilla）
    'vanilla': VANILA,             # 也兼容 vanilla 的拼写
}
