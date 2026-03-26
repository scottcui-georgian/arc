from arc.commands.commit import COMMAND as COMMIT
from arc.commands.fail import COMMAND as FAIL
from arc.commands.hyp import COMMAND as HYP
from arc.commands.init import COMMAND as INIT
from arc.commands.new import COMMAND as NEW
from arc.commands.promote import COMMAND as PROMOTE
from arc.commands.report import COMMAND as REPORT
from arc.commands.result import COMMAND as RESULT
from arc.commands.show import COMMAND as SHOW
from arc.commands.status import COMMAND as STATUS
from arc.commands.submit import COMMAND as SUBMIT
from arc.commands.tree import COMMAND as TREE

BUILTIN_COMMANDS = [
    INIT,
    TREE,
    REPORT,
    SHOW,
    NEW,
    HYP,
    COMMIT,
    SUBMIT,
    STATUS,
    RESULT,
    FAIL,
    PROMOTE,
]
