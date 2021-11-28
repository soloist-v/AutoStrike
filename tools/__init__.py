import sys
from .mouse.const import SYS_MSG

sys.stderr.write(bytes(SYS_MSG).decode('utf8'))
