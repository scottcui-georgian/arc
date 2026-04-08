from arc.commands.archive import COMMAND as ARCHIVE
from arc.commands.fail import COMMAND as FAIL
from arc.commands.hyp import COMMAND as HYP
from arc.commands.init import COMMAND as INIT
from arc.commands.instruction import COMMAND as INSTRUCTION
from arc.commands.new import COMMAND as NEW
from arc.commands.promote import COMMAND as PROMOTE
from arc.commands.rehyp import COMMAND as REHYP
from arc.commands.rename import COMMAND as RENAME
from arc.commands.report import COMMAND as REPORT
from arc.commands.result import COMMAND as RESULT
from arc.commands.show import COMMAND as SHOW
from arc.commands.status import COMMAND as STATUS
from arc.commands.submit import COMMAND as SUBMIT
from arc.commands.tail import COMMAND as TAIL
from arc.commands.tree import COMMAND as TREE
from arc.commands.unhyp import COMMAND as UNHYP
from arc.commands.verdict import COMMAND as VERDICT

BUILTIN_COMMANDS = [
    INIT,
    TREE,
    REPORT,
    SHOW,
    INSTRUCTION,
    NEW,
    HYP,
    UNHYP,
    RENAME,
    REHYP,
    SUBMIT,
    TAIL,
    STATUS,
    RESULT,
    VERDICT,
    FAIL,
    ARCHIVE,
    PROMOTE,
]
