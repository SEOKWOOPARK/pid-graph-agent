import builtins
import pytest
from types import SimpleNamespace
from graph_agent.cli import main
from graph_agent.config import Config
from graph_agent.errors import InvalidInputError, ModelError


def test_help_needs_no_model(capsys):
    with pytest.raises(SystemExit) as exc:
        main(['--help'])
    assert exc.value.code == 0
    assert '--model' in capsys.readouterr().out


def test_invalid_graph_is_controlled(tmp_path, capsys):
    code = main(['--graph', str(tmp_path / 'absent.json'), '--query', 'hello'])
    assert code == 2
    assert 'Error' in capsys.readouterr().err


def test_empty_input_loads_no_model(capsys):
    assert main(['--query', ' ']) == 2
    assert 'nonempty' in capsys.readouterr().err


def test_one_shot_model_override(monkeypatch, capsys):
    async def ask(self, query):
        assert self.config.model_id == 'Qwen/Qwen3-0.6B'
        assert query == 'test'
        return SimpleNamespace(text='grounded answer')
    monkeypatch.setattr('graph_agent.agent.AgentSession.ask', ask)
    assert main(['--model', 'Qwen/Qwen3-0.6B', '--query', 'test']) == 0
    assert capsys.readouterr().out.strip() == 'grounded answer'


def test_interactive_recovers_from_inference_error(monkeypatch, capsys):
    questions = iter(['bad query', 'next query', 'exit'])
    monkeypatch.setattr(builtins, 'input', lambda _: next(questions))
    async def ask(self, query):
        if query == 'bad query':
            raise ModelError('model failure')
        return SimpleNamespace(text='second answer')
    monkeypatch.setattr('graph_agent.agent.AgentSession.ask', ask)
    assert main([]) == 0
    captured = capsys.readouterr()
    assert 'model failure' in captured.err
    assert 'second answer' in captured.out


@pytest.mark.parametrize('kwargs', [
    {'device':'cloud'}, {'dtype':'int8'}, {'max_new_tokens':0},
    {'max_new_tokens':True}, {'max_input_tokens':32769},
    {'max_turns':31}, {'model_id':' '}, {'local_files_only':1},
    {'device': []}, {'dtype': []},
])
def test_invalid_config(kwargs):
    with pytest.raises(InvalidInputError):
        Config(**kwargs)


def test_environment_configuration(monkeypatch):
    monkeypatch.setenv('QWEN_MODEL_ID', 'Qwen/Qwen3-0.6B')
    monkeypatch.setenv('QWEN_DEVICE', 'cpu')
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    config = Config.from_env()
    assert config.model_id == 'Qwen/Qwen3-0.6B' and config.device == 'cpu'
    assert config.local_files_only
    monkeypatch.setenv('QWEN_MODEL_ID', '')
    assert Config.from_env('Qwen/Qwen3-1.7B').model_id == 'Qwen/Qwen3-1.7B'
    monkeypatch.setenv('QWEN_MODEL_ID', 'Qwen/Qwen3-0.6B')
    monkeypatch.setenv('MAX_NEW_TOKENS', 'bad')
    with pytest.raises(InvalidInputError):
        Config.from_env()
