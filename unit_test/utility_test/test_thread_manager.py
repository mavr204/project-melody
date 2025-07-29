import pytest
from unittest.mock import MagicMock, patch
import utility.errors as err 
from utility.thread_manager import ThreadManager, ThreadEvent, ThreadStatus

@pytest.fixture
def thread_manager():
    return ThreadManager()

def test_create_new_thread(thread_manager):
    with patch("utility.thread_manager.Thread") as thread_class, \
         patch("utility.thread_manager.Event") as event_class, \
         patch("utility.thread_manager.ThreadEvent") as thread_event_class:
        
        mock_thread = MagicMock()
        mock_event = MagicMock()
        mock_thread_event = MagicMock()

        thread_class.return_value = mock_thread
        event_class.return_value = mock_event
        thread_event_class.return_value = mock_thread_event

        target = MagicMock()
        name = 'TestThread'

        returned_name = thread_manager.create_new_thread(target=target, name=name)

        assert name == returned_name
        event_class.assert_called_once()
        thread_class.assert_called_once()
        thread_event_class.assert_called_once_with(mock_thread, mock_event)
        
        assert name in thread_manager.active_threads
        assert thread_manager.active_threads[name] == mock_thread_event

def test_create_new_thread_duplicate(thread_manager):
    thread_manager.active_threads['TestThread'] = MagicMock()

    with pytest.raises(err.ThreadAlreadyExistsError):
        thread_manager.create_new_thread(target=MagicMock(), name="TestThread")

def test_create_new_thread_autostart_success(thread_manager):
    target = MagicMock()
    name = "AutostartThread"

    with patch("utility.thread_manager.Thread") as thread_class, \
         patch("utility.thread_manager.Event") as event_class, \
         patch("utility.thread_manager.ThreadEvent") as thread_event_class, \
         patch.object(thread_manager, "start_thread") as start_thread:

        thread_class.return_value = MagicMock()
        event_class.return_value = MagicMock()
        thread_event_class.return_value = MagicMock()

        returned_name = thread_manager.create_new_thread(target=target, name=name, autostart=True)

        assert returned_name == name
        start_thread.assert_called_once_with(name=name)

def test_create_new_thread_autostart_fail_thread_error(thread_manager, caplog):
    target = MagicMock()
    name = "ThreadStartFail"

    with patch("utility.thread_manager.Thread") as thread_class, \
         patch("utility.thread_manager.Event") as event_class, \
         patch("utility.thread_manager.ThreadEvent") as thread_event_class, \
         patch.object(thread_manager, "start_thread", side_effect=err.ThreadError("boom")):

        thread_class.return_value = MagicMock()
        event_class.return_value = MagicMock()
        thread_event_class.return_value = MagicMock()

        returned_name = thread_manager.create_new_thread(target=target, name=name, autostart=True)

        assert returned_name == name
        assert "Failed to start thread" in caplog.text

def test_create_new_thread_autostart_fail_not_found(thread_manager, caplog):
    target = MagicMock()
    name = "ThreadNotFound"

    with patch("utility.thread_manager.Thread") as thread_class, \
         patch("utility.thread_manager.Event") as event_class, \
         patch("utility.thread_manager.ThreadEvent") as thread_event_class, \
         patch.object(thread_manager, "start_thread", side_effect=err.ThreadNotFoundError("not found")) as start_thread, \
         patch.object(thread_manager, "stop_thread") as stop_thread:

        thread_class.return_value = MagicMock()
        event_class.return_value = MagicMock()
        thread_event_class.return_value = MagicMock()

        returned_name = thread_manager.create_new_thread(target=target, name=name, autostart=True)
        
        start_thread.assert_called_once_with(name=name)
        stop_thread.assert_called_once_with(name)
        assert returned_name is None
        assert "There was problem creating the the thread" in caplog.text

def test_stop_thread_success(thread_manager):
    name = "TestThread"
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True
    mock_event = MagicMock()
    mock_event.is_set.return_value = False
    thread_manager.active_threads[name] = ThreadEvent(mock_thread, mock_event)

    thread_manager.stop_thread(name)

    mock_event.set.assert_called_once()
    mock_thread.join.assert_called_once()
    assert name not in thread_manager.active_threads

def test_stop_thread_not_found(thread_manager):
    with pytest.raises(err.ThreadNotFoundError):
        thread_manager.stop_thread("NonExistentThread")

def test_stop_all_threads(thread_manager):
    mock_thread1 = MagicMock()
    mock_thread1.is_alive.return_value = False
    mock_event1 = MagicMock()
    mock_event1.is_set.return_value = False
    thread_manager.active_threads["T1"] = ThreadEvent(mock_thread1, mock_event1)

    mock_thread2 = MagicMock()
    mock_thread2.is_alive.return_value = True
    mock_event2 = MagicMock()
    mock_event2.is_set.return_value = False
    thread_manager.active_threads["T2"] = ThreadEvent(mock_thread2, mock_event2)

    thread_manager.stop_all_threads()

    assert thread_manager.active_threads == {}
    mock_event1.set.assert_called_once()
    mock_event2.set.assert_called_once()
    mock_thread2.join.assert_called_once()

def test_thread_exists_true_and_false(thread_manager):
    mock_thread = MagicMock()
    mock_event = MagicMock()
    thread_manager.active_threads["Thread1"] = ThreadEvent(mock_thread, mock_event)

    assert thread_manager.thread_exists("Thread1") is True
    assert thread_manager.thread_exists("NoThread") is False

def test_get_thread_status_not_found(thread_manager, caplog):
    status = thread_manager.get_thread_status("GhostThread")
    assert status == ThreadStatus.NOT_FOUND
    assert "Thread 'GhostThread' not found." in caplog.text

def test_get_thread_status_running(thread_manager):
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True
    mock_event = MagicMock()
    thread_manager.active_threads["RunningThread"] = ThreadEvent(mock_thread, mock_event)

    assert thread_manager.get_thread_status("RunningThread") == ThreadStatus.RUNNING

def test_get_thread_status_stopped(thread_manager):
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = False
    mock_event = MagicMock()
    mock_event.is_set.return_value = True
    thread_manager.active_threads["StoppedThread"] = ThreadEvent(mock_thread, mock_event)

    assert thread_manager.get_thread_status("StoppedThread") == ThreadStatus.STOPPED

def test_get_thread_status_created(thread_manager):
    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = False
    mock_event = MagicMock()
    mock_event.is_set.return_value = False
    thread_manager.active_threads["CreatedThread"] = ThreadEvent(mock_thread, mock_event)

    assert thread_manager.get_thread_status("CreatedThread") == ThreadStatus.CREATED
