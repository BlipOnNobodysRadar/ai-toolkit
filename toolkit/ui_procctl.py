"""Process ownership for UI-launched runs: launch, per-run supervisor, identity checks, scans, termination
(docs/STATE_MACHINES.md M10). Standard library only, so the UI runs it as `python -I -S toolkit/ui_procctl.py ...`.

    launch    --run-id R --run-dir D --cwd C -- <program> <args...>   (the worker's short-lived relay)
    supervise --run-id R --run-dir D --cwd C -- <program> <args...>   (started detached by `launch`)
    probe | terminate | self                                          (JSON request on stdin, JSON result on stdout)

Run boundary. A run's processes are whatever the OS puts inside one ownership container, never a set derived from
parent links (a reused parent pid or an exited intermediate makes those wrong):
  * Windows: a fresh named job object "Global\\aitk-run-<run id>" (one namespace for every session) created by the
    run's supervisor with KILL_ON_JOB_CLOSE and no breakaway allowed. The root is created already inside it
    (PROC_THREAD_ATTRIBUTE_JOB_LIST), so every descendant is a member from its first instruction. The supervisor
    writes the exit receipt only after the kernel's own count (ActiveProcesses) is 0. If the supervisor dies, closing
    its handle makes the kernel terminate every member; the name disappears at once (measured: before the members
    finished terminating), so the members it recorded are then checked individually.
  * Linux (source only on this repository's Windows test host): the supervisor is a child subreaper; every orphan in
    the subtree is re-parented to it, and the exit receipt is written only when waitpid reports no children at all.
    Without that receipt (supervisor died) there is no proof and the run stays owned.
  * Other platforms: no boundary is implemented, so nothing is launched (the supervisor refuses before spawning).

Boundary claim. Before creating anything the supervisor publishes its claim on the run by hard-linking a complete
record to .run.<id>.boundary.json (a link never replaces an existing file). A reconciler that has to give up an
unattached run links a fence record to the same name instead. Exactly one of them wins, so after a successful fence no
supervisor can ever start a root for that run.

Boot identity: Windows BootId (a per-boot counter kept by the OS), Linux /proc/sys/kernel/random/boot_id. A run or
legacy row recorded under another boot id cannot have a live process.

Process identity is "<pid>@<creation>" (Windows: exact 64-bit FILETIME as decimal; Linux: boot id and start ticks).
A pid that now shows another creation time is a different process: the recorded one exited. Unknown is never exited.
"""
from functools import lru_cache
import json
import os
import subprocess
import sys
import time

IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")

# relay (launch) and supervisor exit codes, read by the worker together with the receipts
EXIT_SPAWNED = 0
EXIT_NOT_SPAWNED = 2       # the program was never started (receipt state "failed")
EXIT_RECEIPT_FAILED = 3    # started, receipt not written; every process of the run was terminated and seen gone
EXIT_UNCERTAIN = 4         # started, receipt not written, and the run could not be confirmed gone

FAULT = os.environ.get("AITK_PROCCTL_FAULT", "")  # test hooks only (ui/tests/launch.test.cjs)
GATE = os.environ.get("AITK_PROCCTL_GATE", "")


def _identity(pid, creation):
    return f"{pid}@{creation}"


def _split_identity(identity):
    if not identity or "@" not in identity:
        return None, None
    pid, creation = identity.split("@", 1)
    try:
        return int(pid), creation
    except ValueError:
        return None, None


def run_files(run_dir, run_id):
    return {
        "receipt": os.path.join(run_dir, f".run.{run_id}.receipt.json"),
        "members": os.path.join(run_dir, f".run.{run_id}.members.json"),
        "exit": os.path.join(run_dir, f".run.{run_id}.exit.json"),
        "boundary": os.path.join(run_dir, f".run.{run_id}.boundary.json"),
    }


def job_name(run_id):
    return f"Global\\aitk-run-{run_id}"


def write_receipt(path, record):
    """Atomic and durable: temp file, fsync, replace (and fsync the directory where the platform supports it)."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    if not IS_WINDOWS:
        try:
            dfd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass


def claim_boundary(path, record):
    """(won, record_on_disk). Publishes a complete record under path only if nothing is there yet (hard link: atomic,
    never replaces). A file system without hard links fails loudly (OSError), it is never emulated."""
    tmp = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
        return True, record
    except FileExistsError:
        return False, _read_json(path)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _wait_gate():
    while GATE and not os.path.exists(GATE):
        time.sleep(0.05)


# ================================================================ Windows
if IS_WINDOWS:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll")

    PROCESS_TERMINATE = 0x0001
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    SYNCHRONIZE = 0x00100000
    JOB_OBJECT_QUERY = 0x0004
    JOB_OBJECT_TERMINATE = 0x0008
    WAIT_OBJECT_0 = 0
    WAIT_TIMEOUT = 0x102
    ERROR_FILE_NOT_FOUND = 2
    ERROR_ACCESS_DENIED = 5
    ERROR_NO_MORE_FILES = 18
    ERROR_INVALID_PARAMETER = 87
    ERROR_MORE_DATA = 234
    ERROR_ALREADY_EXISTS = 183
    TH32CS_SNAPPROCESS = 0x2
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    ProcessCommandLineInformation = 60
    STATUS_INFO_LENGTH_MISMATCH = 0xC0000004
    JobObjectBasicAccountingInformation = 1
    JobObjectBasicProcessIdList = 3
    JobObjectAssociateCompletionPortInformation = 7
    JobObjectExtendedLimitInformation = 9
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    JOB_OBJECT_MSG_NEW_PROCESS = 6
    EXTENDED_STARTUPINFO_PRESENT = 0x00080000
    CREATE_UNICODE_ENVIRONMENT = 0x00000400
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    STARTF_USESTDHANDLES = 0x100
    STARTF_USESHOWWINDOW = 0x1
    PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x20002
    PROC_THREAD_ATTRIBUTE_JOB_LIST = 0x2000D

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260),
        ]

    class UNICODE_STRING(ctypes.Structure):
        _fields_ = [("Length", ctypes.c_ushort), ("MaximumLength", ctypes.c_ushort), ("Buffer", ctypes.c_void_p)]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("Read", "Write", "Other", "ReadT", "WriteT", "OtherT")]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION), ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
            ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
            ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    class JOBOBJECT_ASSOCIATE_COMPLETION_PORT(ctypes.Structure):
        _fields_ = [("CompletionKey", ctypes.c_void_p), ("CompletionPort", wintypes.HANDLE)]

    class STARTUPINFOW(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR), ("lpDesktop", wintypes.LPWSTR),
            ("lpTitle", wintypes.LPWSTR), ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
            ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD), ("dwXCountChars", wintypes.DWORD),
            ("dwYCountChars", wintypes.DWORD), ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
            ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD), ("lpReserved2", ctypes.c_void_p),
            ("hStdInput", wintypes.HANDLE), ("hStdOutput", wintypes.HANDLE), ("hStdError", wintypes.HANDLE),
        ]

    class STARTUPINFOEXW(ctypes.Structure):
        _fields_ = [("StartupInfo", STARTUPINFOW), ("lpAttributeList", ctypes.c_void_p)]

    class PROCESS_INFORMATION(ctypes.Structure):
        _fields_ = [("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
                    ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD)]

    def _fn(dll, name, restype, *argtypes):
        f = getattr(dll, name)
        f.restype, f.argtypes = restype, list(argtypes)
        return f

    H, D, B, P = wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL, ctypes.c_void_p
    OpenProcess = _fn(_k32, "OpenProcess", H, D, B, D)
    CloseHandle = _fn(_k32, "CloseHandle", B, H)
    GetProcessTimes = _fn(_k32, "GetProcessTimes", B, H, *[ctypes.POINTER(ctypes.c_ulonglong)] * 4)
    WaitForSingleObject = _fn(_k32, "WaitForSingleObject", D, H, D)
    CreateToolhelp32Snapshot = _fn(_k32, "CreateToolhelp32Snapshot", H, D, D)
    Process32FirstW = _fn(_k32, "Process32FirstW", B, H, ctypes.POINTER(PROCESSENTRY32W))
    Process32NextW = _fn(_k32, "Process32NextW", B, H, ctypes.POINTER(PROCESSENTRY32W))
    NtQueryInformationProcess = _fn(_ntdll, "NtQueryInformationProcess", ctypes.c_ulong, H, ctypes.c_ulong, P,
                                    ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong))
    CreateJobObjectW = _fn(_k32, "CreateJobObjectW", H, P, wintypes.LPCWSTR)
    OpenJobObjectW = _fn(_k32, "OpenJobObjectW", H, D, B, wintypes.LPCWSTR)
    SetInformationJobObject = _fn(_k32, "SetInformationJobObject", B, H, ctypes.c_int, P, D)
    QueryInformationJobObject = _fn(_k32, "QueryInformationJobObject", B, H, ctypes.c_int, P, D, ctypes.POINTER(D))
    TerminateJobObject = _fn(_k32, "TerminateJobObject", B, H, wintypes.UINT)
    CreateIoCompletionPort = _fn(_k32, "CreateIoCompletionPort", H, H, H, ctypes.c_size_t, D)
    GetQueuedCompletionStatus = _fn(_k32, "GetQueuedCompletionStatus", B, H, ctypes.POINTER(D),
                                    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_void_p), D)
    InitializeProcThreadAttributeList = _fn(_k32, "InitializeProcThreadAttributeList", B, P, D, D,
                                            ctypes.POINTER(ctypes.c_size_t))
    UpdateProcThreadAttribute = _fn(_k32, "UpdateProcThreadAttribute", B, P, D, ctypes.c_size_t, P, ctypes.c_size_t,
                                    P, P)
    DeleteProcThreadAttributeList = _fn(_k32, "DeleteProcThreadAttributeList", None, P)
    CreateProcessW = _fn(_k32, "CreateProcessW", B, wintypes.LPCWSTR, wintypes.LPWSTR, P, P, B, D, P,
                         wintypes.LPCWSTR, ctypes.POINTER(STARTUPINFOEXW), ctypes.POINTER(PROCESS_INFORMATION))
    GetTickCount64 = _fn(_k32, "GetTickCount64", ctypes.c_ulonglong)

    def _creation(h):
        c, e, k, u = (ctypes.c_ulonglong() for _ in range(4))
        if not GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u)):
            return None
        return str(c.value)

    def _alive(h):
        """True (running), False (exited: the handle is signaled), None (the wait failed: unknown)."""
        r = WaitForSingleObject(h, 0)
        if r == WAIT_TIMEOUT:
            return True
        if r == WAIT_OBJECT_0:
            return False
        return None

    def _cmdline(h):
        size = ctypes.c_ulong(0)
        status = NtQueryInformationProcess(h, ProcessCommandLineInformation, None, 0, ctypes.byref(size))
        if status != STATUS_INFO_LENGTH_MISMATCH or size.value == 0:
            return None
        buf = ctypes.create_string_buffer(size.value)
        if NtQueryInformationProcess(h, ProcessCommandLineInformation, buf, size, ctypes.byref(size)) != 0:
            return None
        us = UNICODE_STRING.from_buffer(buf)
        return ctypes.wstring_at(us.Buffer, us.Length // 2) if us.Buffer else ""

    def _open(pid, access=PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE):
        """(handle, None) or (None, 'gone' | 'denied' | 'error N')."""
        h = OpenProcess(access, False, int(pid))
        if h:
            return h, None
        err = ctypes.get_last_error()
        if err == ERROR_INVALID_PARAMETER:
            return None, "gone"
        if err == ERROR_ACCESS_DENIED:
            return None, "denied"
        return None, f"error {err}"

    def _snapshot():
        """[(pid, ppid, executable name)] from one Toolhelp snapshot; raises unless enumeration reached its normal end."""
        snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if not snap or snap == INVALID_HANDLE_VALUE:
            raise OSError(f"CreateToolhelp32Snapshot failed ({ctypes.get_last_error()})")
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            out = []
            ok = Process32FirstW(snap, ctypes.byref(entry))
            while ok:
                out.append((int(entry.th32ProcessID), int(entry.th32ParentProcessID), entry.szExeFile))
                ok = Process32NextW(snap, ctypes.byref(entry))
            err = ctypes.get_last_error()
            if err != ERROR_NO_MORE_FILES:
                raise OSError(f"process enumeration stopped early ({err})")
            return out
        finally:
            CloseHandle(snap)

    def _job_open(run_id, access=JOB_OBJECT_QUERY):
        h = OpenJobObjectW(access, False, job_name(run_id))
        if h:
            return h, None
        err = ctypes.get_last_error()
        return None, ("absent" if err == ERROR_FILE_NOT_FOUND else f"error {err}")

    def _job_active(job):
        info = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        if not QueryInformationJobObject(job, JobObjectBasicAccountingInformation, ctypes.byref(info),
                                         ctypes.sizeof(info), None):
            raise OSError(f"QueryInformationJobObject failed ({ctypes.get_last_error()})")
        return int(info.ActiveProcesses)

    def _job_pids(job):
        n = 256
        while True:
            class LIST(ctypes.Structure):
                _fields_ = [("Assigned", wintypes.DWORD), ("InList", wintypes.DWORD),
                            ("Ids", ctypes.c_size_t * n)]
            lst = LIST()
            if QueryInformationJobObject(job, JobObjectBasicProcessIdList, ctypes.byref(lst), ctypes.sizeof(lst),
                                         None):
                return [int(lst.Ids[i]) for i in range(lst.InList)]
            if ctypes.get_last_error() != ERROR_MORE_DATA:
                raise OSError(f"QueryInformationJobObject(pid list) failed ({ctypes.get_last_error()})")
            n *= 4

    def _boot_ms():
        return int(time.time() * 1000) - int(GetTickCount64())

# ================================================================ Linux
if IS_LINUX:
    import signal as _signal

    def _read_boot_id():
        try:
            with open("/proc/sys/kernel/random/boot_id") as f:
                return f.read().strip()
        except OSError:
            return "noboot"

    _BOOT = _read_boot_id()

    def _stat(pid):
        """(state, ppid, starttime) or None if there is no such process."""
        try:
            with open(f"/proc/{int(pid)}/stat", "rb") as f:
                raw = f.read().decode("utf-8", "replace")
        except (FileNotFoundError, ProcessLookupError):
            return None
        fields = raw[raw.rindex(")") + 2:].split()
        return fields[0], int(fields[1]), fields[19]

    def _cmdline_linux(pid):
        try:
            with open(f"/proc/{int(pid)}/cmdline", "rb") as f:
                return f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            return None

    @lru_cache(maxsize=1)
    def _btime():
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("btime "):
                    return int(line.split()[1])
        return 0

    _CLK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100

    def _boot_ms():
        return _btime() * 1000


def platform_name():
    return "windows" if IS_WINDOWS else ("linux" if IS_LINUX else sys.platform)


def boot_ms():
    try:
        return _boot_ms()
    except Exception:  # noqa: BLE001
        return None


def boot_id():
    """Identity of the current boot, or None if it cannot be read (then no reboot proof is possible)."""
    try:
        if IS_WINDOWS:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management\PrefetchParameters")
            try:
                value, _kind = winreg.QueryValueEx(key, "BootId")
            finally:
                winreg.CloseKey(key)
            return f"win-boot:{int(value)}"
        if IS_LINUX:
            return f"linux-boot:{_BOOT}" if _BOOT != "noboot" else None
    except Exception:  # noqa: BLE001
        return None
    return None


# ================================================================ identity / liveness
def describe(pid):
    """The process currently at pid: {state: alive|gone|denied|unknown, identity?, cmdline?}."""
    pid = int(pid)
    if IS_WINDOWS:
        h, why = _open(pid)
        if h is None:
            return {"state": {"gone": "gone", "denied": "denied"}.get(why, "unknown"), "why": why}
        try:
            creation = _creation(h)
            if creation is None:
                return {"state": "unknown", "why": "no creation time"}
            alive = _alive(h)
            if alive is None:
                return {"state": "unknown", "why": "wait failed"}
            if not alive:
                return {"state": "gone", "why": "exited", "identity": _identity(pid, creation)}
            return {"state": "alive", "identity": _identity(pid, creation),
                    "started_ms": int(creation) // 10000 - 11644473600000, "cmdline": _cmdline(h)}
        finally:
            CloseHandle(h)
    if IS_LINUX:
        st = _stat(pid)
        if st is None or st[0] == "Z":
            return {"state": "gone", "why": "no process" if st is None else "zombie"}
        return {"state": "alive", "identity": _identity(pid, f"{_BOOT}:{st[2]}"),
                "started_ms": int((_btime() + int(st[2]) / _CLK) * 1000), "cmdline": _cmdline_linux(pid)}
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-o", "command=", "-p", str(pid)], capture_output=True, text=True,
                             timeout=10)
    except Exception as e:  # noqa: BLE001 - any failure is "unknown", never "gone"
        return {"state": "unknown", "why": str(e)}
    line = out.stdout.strip()
    if out.returncode == 1 and not line:
        return {"state": "gone", "why": "no process"}
    if out.returncode != 0 or not line:
        return {"state": "unknown", "why": f"ps rc {out.returncode}"}
    return {"state": "alive", "identity": _identity(pid, line[:24].strip().replace(" ", "_")), "cmdline": line[24:]}


def check(identity):
    """'alive' (that very process still runs), 'exited' (no such pid, or the pid shows another creation time),
    or 'unknown' (anything that could not be established)."""
    pid, creation = _split_identity(identity)
    if pid is None or not creation or creation == "None":
        return "unknown"
    d = describe(pid)
    if d["state"] == "gone":
        return "exited"
    if d["state"] == "alive":
        return "alive" if d["identity"] == identity else "exited"
    return "unknown"


def scan(needle, since_ms, legacy_launchers_only=False):
    """Live processes created at/after since_ms whose command line contains needle.
    complete=False if any such process could be opened but its creation time or command line could not be read.
    A process that cannot be queried at all ('denied') also makes the scan incomplete.
    Legacy launch scans on Windows consider Python and its Node relay."""
    matches, unreadable, denied = [], 0, 0
    if IS_WINDOWS:
        since_ft = (int(since_ms) + 11644473600000) * 10000
        for pid, ppid, executable in _snapshot():
            if legacy_launchers_only and not (executable.lower().startswith("python") or executable.lower() == "node.exe"):
                continue
            if pid in (0, 4):
                continue
            h, why = _open(pid)
            if h is None:
                if why == "denied":
                    denied += 1
                if why != "gone":
                    unreadable += 1
                continue
            try:
                creation = _creation(h)
                alive = _alive(h)
                if creation is None or alive is None:
                    unreadable += 1
                    continue
                if int(creation) < since_ft or not alive:
                    continue
                cmd = _cmdline(h)
                if cmd is None:
                    unreadable += 1
                elif needle in cmd:
                    matches.append({"identity": _identity(pid, creation), "ppid": ppid, "cmdline": cmd})
            finally:
                CloseHandle(h)
        return {"complete": unreadable == 0, "matches": matches, "unreadable": unreadable, "denied": denied}
    if IS_LINUX:
        boot_since = (since_ms / 1000.0 - _btime()) * _CLK
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            st = _stat(int(name))
            if st is None or st[0] == "Z" or int(st[2]) < boot_since:
                continue
            cmd = _cmdline_linux(int(name))
            if cmd is None:
                unreadable += 1
            elif needle in cmd:
                matches.append({"identity": _identity(int(name), f"{_BOOT}:{st[2]}"), "ppid": st[1], "cmdline": cmd})
        return {"complete": unreadable == 0, "matches": matches, "unreadable": unreadable, "denied": denied}
    return {"complete": False, "matches": [], "unreadable": -1, "why": "process scan unsupported on this platform"}


# ================================================================ run boundary state
def run_state(run_id, supervisor, recorded):
    """{supervisor: alive|exited|unknown, boundary: open|absent|unknown, active?, live: [...], members: {id: state}}"""
    out = {"supervisor": check(supervisor) if supervisor else "unknown", "live": [], "members": {}}
    for ident in recorded:
        out["members"][ident] = check(ident)
    if IS_WINDOWS:
        job, why = _job_open(run_id)
        if job is None:
            out["boundary"] = "absent" if why == "absent" else "unknown"
            if why != "absent":
                out["error"] = why
            return out
        try:
            out["active"] = _job_active(job)
            for pid in _job_pids(job):
                d = describe(pid)
                if d["state"] == "alive":
                    out["live"].append(d["identity"])
            out["boundary"] = "open"
        except OSError as e:
            out["boundary"] = "unknown"
            out["error"] = str(e)
        finally:
            CloseHandle(job)
        return out
    out["boundary"] = "unknown"  # POSIX: only the supervisor's own exit receipt proves the subtree empty
    return out


# ================================================================ termination
def terminate(run_id, supervisor=None, wait_ms=15000):
    """Force stop every process of a run through its ownership container. {"supported", "verified", ...}"""
    if IS_WINDOWS:
        job, why = _job_open(run_id, JOB_OBJECT_QUERY | JOB_OBJECT_TERMINATE)
        if job is None:
            return {"supported": True, "verified": False, "boundary": "absent" if why == "absent" else "unknown",
                    "error": f"the run's job object could not be opened ({why})"}
        try:
            if not TerminateJobObject(job, 1):
                return {"supported": True, "verified": False, "boundary": "open",
                        "error": f"TerminateJobObject failed ({ctypes.get_last_error()})"}
            # termination is asynchronous: only the kernel's count reaching 0 means the members are gone
            deadline = time.monotonic() + wait_ms / 1000.0
            active = _job_active(job)
            while active and time.monotonic() < deadline:
                time.sleep(0.05)
                active = _job_active(job)
            return {"supported": True, "verified": active == 0, "boundary": "open", "remaining": active}
        finally:
            CloseHandle(job)
    if IS_LINUX and hasattr(os, "pidfd_open") and hasattr(_signal, "pidfd_send_signal") and supervisor:
        # pragma: no cover - not executed on the Windows test host. Ask the run's supervisor (identity-checked
        # through a pidfd) to kill its subtree; exit is proved only by its later exit receipt.
        pid, creation = _split_identity(supervisor)
        try:
            fd = os.pidfd_open(pid)
        except OSError as e:
            return {"supported": True, "verified": False, "error": f"supervisor not reachable ({e})"}
        try:
            st = _stat(pid)
            _signal.pidfd_send_signal(fd, 0)
            if st is None or f"{_BOOT}:{st[2]}" != creation:
                return {"supported": True, "verified": False, "error": "supervisor identity changed"}
            _signal.pidfd_send_signal(fd, _signal.SIGUSR1)
            return {"supported": True, "verified": False, "requested": True}
        finally:
            os.close(fd)
    return {"supported": False, "verified": False,
            "error": f"force stop with a verified process boundary is not supported on {sys.platform}; the stop "
                     "request stays pending and the run keeps its GPUs until its processes are proven gone"}


# ================================================================ launch (relay) and supervisor
def _hidden_popen_kwargs():
    kw = {"stdin": subprocess.DEVNULL, "stdout": 1, "stderr": 2, "close_fds": True}
    if IS_WINDOWS:
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0  # SW_HIDE
        kw["startupinfo"] = si
        kw["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    return kw


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def launch(run_id, run_dir, cwd, program):
    """The worker's relay: start the supervisor detached (outside the worker's process tree) and wait for its spawn
    receipt or its exit, however long that takes. Exit code mirrors the outcome."""
    files = run_files(run_dir, run_id)
    me = os.path.abspath(__file__)
    # the base interpreter (stdlib only is needed): a venv launcher would add a process layer around the supervisor
    python = getattr(sys, "_base_executable", None) or sys.executable
    argv = [python, "-I", "-S", me, "supervise", "--run-id", run_id, "--run-dir", run_dir, "--cwd", cwd, "--",
            *program]
    try:
        sup = subprocess.Popen(argv, cwd=cwd, **_hidden_popen_kwargs())
    except OSError as e:
        try:
            write_receipt(files["receipt"], {"run_id": run_id, "state": "failed", "error": f"supervisor: {e}"})
        except OSError:
            pass
        sys.stderr.write(f"Error launching job supervisor: {e}\n")
        return EXIT_NOT_SPAWNED
    while True:
        r = _read_json(files["receipt"])
        if r and r.get("run_id") == run_id and r.get("state") in ("spawned", "failed"):
            return EXIT_SPAWNED if r["state"] == "spawned" else EXIT_NOT_SPAWNED
        code = sup.poll()  # our own child: the handle Popen holds, never a bare pid
        if code is not None:
            r = _read_json(files["receipt"])
            if r and r.get("run_id") == run_id and r.get("state") in ("spawned", "failed"):
                return EXIT_SPAWNED if r["state"] == "spawned" else EXIT_NOT_SPAWNED
            return code if code in (EXIT_NOT_SPAWNED, EXIT_RECEIPT_FAILED) else EXIT_UNCERTAIN
        time.sleep(0.05)


def _own_identity():
    return describe(os.getpid()).get("identity")


def supervise(run_id, run_dir, cwd, program):
    files = run_files(run_dir, run_id)
    me = _own_identity()
    base = {"run_id": run_id, "supervisor": me, "platform": platform_name(), "boot_id": boot_id()}
    if not (IS_WINDOWS or IS_LINUX):
        return _fail(files, base, f"no process ownership boundary is implemented for {sys.platform}; nothing was started")
    if not me or not base["boot_id"]:
        return _fail(files, base, "the supervisor could not read its own identity or the boot identity")
    if FAULT == "delay_before_claim":  # test hook: a slow supervisor racing a reconciler's fence
        _wait_gate()
    try:
        won, onfile = claim_boundary(files["boundary"], {**base, "kind": "supervisor"})
    except OSError as e:
        return _fail(files, base, f"cannot publish the run's boundary claim ({e})")
    if not won:
        what = (onfile or {}).get("kind", "unreadable")
        return _fail(files, base, f"this run was already claimed ({what}); nothing was started")
    if IS_WINDOWS:
        return _supervise_windows(run_id, files, cwd, program, base)
    return _supervise_linux(run_id, files, cwd, program, base)


def fence(run_id, run_dir, by):
    """Give up an unattached run for good: wins only if no supervisor has claimed it, and then no supervisor ever can."""
    files = run_files(run_dir, run_id)
    try:
        won, onfile = claim_boundary(files["boundary"], {"run_id": run_id, "kind": "fence", "by": by, "boot_id": boot_id()})
    except OSError as e:
        return {"fenced": False, "error": str(e)}
    return {"fenced": won, "boundary": None if won else onfile}


def _fail(files, base, error, code=EXIT_NOT_SPAWNED):
    try:
        write_receipt(files["receipt"], {**base, "state": "failed", "error": error})
    except OSError:
        pass
    sys.stderr.write(f"Error launching job process: {error}\n")
    return code


def _write_members(files, base, members):
    try:
        write_receipt(files["members"], {**base, "members": sorted(members)})
    except OSError as e:
        sys.stderr.write(f"Could not record run members: {e}\n")


def _supervise_windows(run_id, files, cwd, program, base):
    job = CreateJobObjectW(None, job_name(run_id))
    err = ctypes.get_last_error()
    if not job:
        return _fail(files, base, f"CreateJobObject failed ({err})")
    if err == ERROR_ALREADY_EXISTS or FAULT == "job_exists":
        CloseHandle(job)  # someone else's object: never adopt it as this run's boundary
        return _fail(files, base, "a job object for this run id already exists")
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE  # no BREAKAWAY_OK flags
    if not SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)):
        CloseHandle(job)
        return _fail(files, base, f"SetInformationJobObject failed ({ctypes.get_last_error()})")
    port = CreateIoCompletionPort(INVALID_HANDLE_VALUE, None, 0, 1)
    if port:
        assoc = JOBOBJECT_ASSOCIATE_COMPLETION_PORT(None, port)
        SetInformationJobObject(job, JobObjectAssociateCompletionPortInformation, ctypes.byref(assoc),
                                ctypes.sizeof(assoc))
    base = {**base, "boundary": "win-job", "job": job_name(run_id)}
    if FAULT == "before_spawn":
        program = [os.path.join(files["receipt"] + ".missing", "nothing.exe")]
    try:
        root, pid = _create_in_job(job, program, cwd)
    except OSError as e:
        CloseHandle(job)
        return _fail(files, base, f"{type(e).__name__}: {e}")
    members = set()
    try:
        ident = _identity(pid, _creation(root))
        members.add(ident)
        if FAULT == "hang_after_spawn":
            _wait_gate()
        if FAULT == "crash_before_receipt":  # test hook: supervisor dies after the root exists, before any receipt
            _wait_gate()
            os._exit(9)
        if FAULT == "receipt_write":
            raise OSError("injected receipt write failure")
        write_receipt(files["receipt"], {**base, "state": "spawned", "pid": pid, "identity": ident})
    except Exception as e:  # noqa: BLE001 - never leave an unrecorded run running
        sys.stderr.write(f"Could not record the job process ({e}); terminating the run\n")
        TerminateJobObject(job, 1)
        try:
            deadline = time.monotonic() + 30
            while _job_active(job) and time.monotonic() < deadline:
                time.sleep(0.05)
            gone = _job_active(job) == 0
        except OSError:
            gone = False
        CloseHandle(root)
        CloseHandle(job)
        return EXIT_RECEIPT_FAILED if gone else EXIT_UNCERTAIN
    CloseHandle(root)
    _write_members(files, base, members)
    # monitor until the kernel reports no active process in the job
    while True:
        changed = False
        if port:
            msg, key, ov = wintypes.DWORD(), ctypes.c_size_t(), ctypes.c_void_p()
            while GetQueuedCompletionStatus(port, ctypes.byref(msg), ctypes.byref(key), ctypes.byref(ov), 200):
                if msg.value == JOB_OBJECT_MSG_NEW_PROCESS and ov.value:
                    d = describe(int(ov.value))
                    if d.get("identity") and d["identity"] not in members:
                        members.add(d["identity"])
                        changed = True
        else:
            time.sleep(0.2)
        try:
            for p in _job_pids(job):  # completion messages are not guaranteed; the pid list is
                d = describe(p)
                if d.get("identity") and d["identity"] not in members:
                    members.add(d["identity"])
                    changed = True
            if changed:
                _write_members(files, base, members)
            if _job_active(job) == 0:
                break
        except OSError as e:
            sys.stderr.write(f"Run supervisor: job query failed ({e}); retrying\n")
            time.sleep(1)
    _write_members(files, base, members)
    write_receipt(files["exit"], {**base, "state": "exited", "members": sorted(members), "evidence":
                                  "job object ActiveProcesses == 0"})
    CloseHandle(job)
    return 0


def _create_in_job(job, program, cwd):
    """CreateProcessW with the job in PROC_THREAD_ATTRIBUTE_JOB_LIST: the process belongs to the job before it runs.
    stdio: this supervisor's stdout/stderr (the job log) and NUL; only those handles are inherited."""
    nul = open(os.devnull, "rb")
    try:
        hin = msvcrt.get_osfhandle(nul.fileno())
        hout = msvcrt.get_osfhandle(1)
        herr = msvcrt.get_osfhandle(2)
        handles = []
        for h in (hin, hout, herr):
            os.set_handle_inheritable(h, True)
            if h not in handles:
                handles.append(h)
        size = ctypes.c_size_t(0)
        InitializeProcThreadAttributeList(None, 2, 0, ctypes.byref(size))
        attrs = ctypes.create_string_buffer(size.value)
        if not InitializeProcThreadAttributeList(attrs, 2, 0, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            job_arr = (wintypes.HANDLE * 1)(job)
            h_arr = (wintypes.HANDLE * len(handles))(*handles)
            if not UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_JOB_LIST, job_arr,
                                             ctypes.sizeof(job_arr), None, None):
                raise ctypes.WinError(ctypes.get_last_error())
            if not UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, h_arr,
                                             ctypes.sizeof(h_arr), None, None):
                raise ctypes.WinError(ctypes.get_last_error())
            si = STARTUPINFOEXW()
            si.StartupInfo.cb = ctypes.sizeof(STARTUPINFOEXW)
            si.StartupInfo.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW
            si.StartupInfo.wShowWindow = 0
            si.StartupInfo.hStdInput, si.StartupInfo.hStdOutput, si.StartupInfo.hStdError = hin, hout, herr
            si.lpAttributeList = ctypes.cast(attrs, ctypes.c_void_p)
            pi = PROCESS_INFORMATION()
            cmd = ctypes.create_unicode_buffer(subprocess.list2cmdline(program))
            flags = EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
            if not CreateProcessW(None, cmd, None, None, True, flags, None, cwd, ctypes.byref(si), ctypes.byref(pi)):
                raise ctypes.WinError(ctypes.get_last_error())
            CloseHandle(pi.hThread)
            return pi.hProcess, int(pi.dwProcessId)
        finally:
            DeleteProcThreadAttributeList(attrs)
    finally:
        nul.close()


def _supervise_linux(run_id, files, cwd, program, base):  # pragma: no cover - not executed on the Windows test host
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if not (hasattr(os, "pidfd_open") and hasattr(_signal, "pidfd_send_signal")):
        return _fail(files, base, "pidfd is not available (Linux 5.3+ and Python 3.9+ are required); nothing was started")
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        return _fail(files, base, f"PR_SET_CHILD_SUBREAPER failed (errno {ctypes.get_errno()})")
    base = {**base, "boundary": "linux-subreaper"}
    kill_requested = []
    _signal.signal(_signal.SIGUSR1, lambda *_: kill_requested.append(True))
    try:
        proc = subprocess.Popen(program, cwd=cwd, **_hidden_popen_kwargs())
    except OSError as e:
        return _fail(files, base, f"{type(e).__name__}: {e}")
    st = _stat(proc.pid)  # our unreaped child: the pid cannot have been reused
    ident = _identity(proc.pid, f"{_BOOT}:{st[2]}") if st else None
    try:
        if ident is None:
            raise OSError("child identity unreadable")
        write_receipt(files["receipt"], {**base, "state": "spawned", "pid": proc.pid, "identity": ident})
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"Could not record the job process ({e}); terminating the run\n")
        kill_requested.append(True)
    while True:
        if kill_requested:
            _kill_children_linux(os.getpid())
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break  # no children at all: as subreaper, the whole subtree is gone
        if pid == 0:
            time.sleep(0.1)
    if not os.path.exists(files["receipt"]):
        return EXIT_RECEIPT_FAILED
    write_receipt(files["exit"], {**base, "state": "exited", "evidence": "subreaper waitpid ECHILD"})
    return 0


def direct_children(supervisor_pid, table):
    """Pids whose CURRENT parent is the live supervisor, from (pid, ppid) rows. As child subreaper the supervisor
    adopts every orphan of its subtree, so repeating kill + reap over its direct children reaches every descendant
    without inferring ownership from older ancestry (a reused intermediate pid cannot redirect it)."""
    return [pid for pid, ppid in table if ppid == supervisor_pid and pid != supervisor_pid]


def _kill_children_linux(supervisor_pid):  # pragma: no cover - not executed on the Windows test host
    table = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            st = _stat(int(name))
            if st is not None:
                table.append((int(name), st[1]))
    for pid in direct_children(supervisor_pid, table):
        try:
            fd = os.pidfd_open(pid)
        except OSError:
            continue
        try:
            st = _stat(pid)  # re-read while the pidfd pins the process: still our child?
            _signal.pidfd_send_signal(fd, 0)
            if st is not None and st[1] == supervisor_pid:
                _signal.pidfd_send_signal(fd, _signal.SIGKILL)
        except OSError:
            pass
        finally:
            os.close(fd)


# ================================================================ CLI
def _parse_run_args(args):
    if "--" not in args:
        raise SystemExit("missing -- before the program")
    sep = args.index("--")
    opts, program = args[:sep], args[sep + 1:]
    o = dict(zip(opts[::2], opts[1::2]))
    return o["--run-id"], o["--run-dir"], o.get("--cwd") or os.getcwd(), program


def main(argv):
    if not argv:
        sys.stderr.write(__doc__)
        return 64
    cmd = argv[0]
    if cmd in ("launch", "supervise"):
        run_id, run_dir, cwd, program = _parse_run_args(argv[1:])
        return (launch if cmd == "launch" else supervise)(run_id, run_dir, cwd, program)
    req = json.loads(sys.stdin.read() or "{}")
    if cmd == "probe":
        res = {"platform": platform_name(), "boot_id": boot_id(), "describe": {}, "runs": {}, "scans": {}}
        for p in req.get("describe", []):
            try:
                res["describe"][str(p)] = describe(p)
            except OSError as e:
                res["describe"][str(p)] = {"state": "unknown", "why": str(e)}
        for key, r in (req.get("runs") or {}).items():
            try:
                res["runs"][key] = run_state(r["run_id"], r.get("supervisor"), r.get("members") or [])
            except Exception as e:  # noqa: BLE001
                res["runs"][key] = {"supervisor": "unknown", "boundary": "unknown", "live": [], "members": {},
                                    "error": str(e)}
        for key, s in (req.get("scans") or {}).items():
            try:
                res["scans"][key] = scan(s["needle"], s["since_ms"], s.get("legacy_launchers_only", False))
            except Exception as e:  # noqa: BLE001
                res["scans"][key] = {"complete": False, "matches": [], "error": str(e)}
    elif cmd == "fence":
        res = fence(req["run_id"], req["run_dir"], req.get("by"))
    elif cmd == "terminate":
        res = terminate(req["run_id"], req.get("supervisor"), int(req.get("wait_ms", 15000)))
    elif cmd == "self":
        res = {"platform": platform_name(), "identity": describe(int(req["pid"])), "boot_id": boot_id()}
    else:
        sys.stderr.write(f"unknown command {cmd}\n")
        return 64
    sys.stdout.write(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
