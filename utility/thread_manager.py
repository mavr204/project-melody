from dataclasses import dataclass
from threading import Thread, Event
from utility.logger import get_logger
import utility.errors as err
from enum import Enum

class ThreadStatus(Enum):
    NOT_FOUND = "not_found"
    RUNNING = "running"
    STOPPED = "stopped"
    CREATED = "created"

logger = get_logger(__name__)

@dataclass
class ThreadEvent:
    thread: Thread
    stop_event: Event

class ThreadManager:
    def __init__(self) -> None:
        self.active_threads: dict[str, ThreadEvent] = {}

    def create_new_thread(self, target: callable, args: tuple, name: str, autostart=False) -> str | None:
        t = Thread(target=target, args=args, name=name)
        stop_event = Event()
        self.active_threads[name] = ThreadEvent(t, stop_event)

        if autostart:
            try:
                self.start_thread(name=name)
                logger.info(f'Thread: {name} started')
            except err.ThreadError as e:
                logger.critical(f"Failed to start thread: {name}\nError:{e}")
            except err.ThreadNotFoundError as e:
                logger.critical(f"There was problem creating the the thread: {name}\nError:{e}")
                self.stop_thread(name)
                return None

        return name

    def start_thread(self, name:str) -> None:
        t = self.active_threads.get(name)
        if t is None:
            logger.error(f"No {name} thread found")
            raise err.ThreadNotFoundError(f"{name} thread was not found")
        
        try:
            t.thread.start()
        except RuntimeError as e:
            logger.error(f"Thread {name} could not be started: {e}")
            raise err.ThreadError from e

    def stop_thread(self, name: str) -> None:
        t = self.active_threads.get(name)

        if t is None:
            logger.error(f"No {name} thread found")
            raise err.ThreadNotFoundError(f"{name} thread was not found")

        if not t.stop_event.is_set():
            t.stop_event.set()

        if t.thread.is_alive():
            t.thread.join()
        
        del self.active_threads[name]

    def stop_all_threads(self) -> None:
        for name in list(self.active_threads.keys()):
            try:
                self.stop_thread(name)
            except err.ThreadNotFoundError as e:
                logger.error(f"Thread not found: {e}")
                continue

    def thread_exists(self, name: str) -> bool:
        return name in self.active_threads
    
    def get_thread_status(self, name: str) -> ThreadStatus:
        t = self.active_threads.get(name)
        if t is None:
            logger.warning(f"Thread '{name}' not found.")
            return ThreadStatus.NOT_FOUND

        if t.thread.is_alive():
            return ThreadStatus.RUNNING
        elif t.stop_event.is_set():
            return ThreadStatus.STOPPED
        else:
            return ThreadStatus.CREATED