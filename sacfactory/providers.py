"""Provider setup and authenticated model discovery without experiment uploads."""
from __future__ import annotations

import os
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request

from .language import ApiConfig, LanguageError, post_json


MIT_PARLEY_BASE = 'https://parley.api.mit.edu/v1'


def same_base(left, right):
    return left.strip().rstrip('/') == right.strip().rstrip('/')


def environment_config(environ=None):
    """Use one coherent variable family; never mix credentials across providers."""
    env = os.environ if environ is None else environ
    family = 'SACFACTORY' if any(env.get(k) for k in (
        'SACFACTORY_API_BASE_URL', 'SACFACTORY_API_MODEL', 'SACFACTORY_API_KEY',
        'SACFACTORY_API_FORMAT')) else 'OPENAI'
    if family == 'SACFACTORY':
        base = env.get('SACFACTORY_API_BASE_URL') or 'https://api.deepseek.com'
        model = env.get('SACFACTORY_API_MODEL', '')
        key = env.get('SACFACTORY_API_KEY', '')
        fmt = env.get('SACFACTORY_API_FORMAT')
    else:
        active = any(env.get(k) for k in ('OPENAI_BASE_URL', 'OPENAI_API_KEY', 'OPENAI_MODEL', 'OPENAI_MODEL_ID'))
        base = env.get('OPENAI_BASE_URL') or ('https://api.openai.com/v1' if active else 'https://api.deepseek.com')
        model = env.get('OPENAI_MODEL') or env.get('OPENAI_MODEL_ID', '')
        key = env.get('OPENAI_API_KEY', '')
        fmt = env.get('OPENAI_API_FORMAT')
    default_format = 'responses' if same_base(base, MIT_PARLEY_BASE) or same_base(base, 'https://api.openai.com/v1') else 'chat_completions'
    return ApiConfig(base, model, key, fmt or default_format)


def setup_candidate(base, model, entered_key, api_format, pending=None, current=None, environ=None):
    """Retain a session key only for the same endpoint, including incomplete setup."""
    saved = next((c.api_key for c in (pending, current, environment_config(environ))
                  if c and same_base(c.base_url, base) and c.api_key), '')
    return ApiConfig(base, model, entered_key or saved, api_format)


def list_available_models(config, transport=None):
    """GET model metadata only, with no inference call, records or retries."""
    endpoint = config.models_endpoint()
    request = Request(endpoint, headers={'Authorization': 'Bearer ' + config.api_key.strip(),
                                        'Accept': 'application/json'}, method='GET')
    try:
        response = (transport or post_json)(request, 30)
    except HTTPError as exc:
        messages = {
            401: 'The provider rejected the API key (HTTP 401). Check your institutional key.',
            403: 'Model discovery is forbidden for this account (HTTP 403). Check institutional access.',
            404: 'This provider does not expose /models (HTTP 404). Enter the exact model ID from its documentation.',
            429: 'The provider rate or quota limit was reached (HTTP 429). Try later or check your account.',
        }
        raise LanguageError(messages.get(exc.code, 'Model discovery failed with a provider HTTP error.')) from None
    except (URLError, TimeoutError, OSError):
        raise LanguageError('Could not connect to the model catalog. Check the base URL, network or VPN and try again.') from None
    except (ValueError, TypeError):
        raise LanguageError('The provider returned an invalid model catalog. You can enter a model ID manually.') from None
    if not isinstance(response, dict) or not isinstance(response.get('data'), list):
        raise LanguageError('The provider returned an unsupported model catalog. Enter a model ID manually.')
    ids = []
    for item in response['data']:
        model = item.get('id') if isinstance(item, dict) else None
        if (isinstance(model, str) and model.strip() and len(model) <= 256
                and not any(ord(c) < 32 for c in model) and config.api_key not in model):
            ids.append(model.strip())
    if not ids:
        raise LanguageError('No available models were returned. Check your provider account or enter a model ID manually.')
    return sorted(set(ids))[:1000]


def probe_chat_connection(config, transport=None):
    """One minimal inference request; never send experiment or conversation data."""
    endpoint = config.research_endpoint()
    if config.api_format == 'responses':
        body = {'model': config.model, 'input': 'Reply with OK.', 'store': False, 'max_output_tokens': 256}
        route = '/responses'
    else:
        body = {'model': config.model, 'messages': [{'role': 'user', 'content': 'Reply with OK.'}],
                'stream': False, 'max_tokens': 256}
        route = '/chat/completions'
    request = Request(endpoint, data=json.dumps(body).encode('utf-8'),
                      headers={'Authorization': 'Bearer ' + config.api_key, 'Content-Type': 'application/json'}, method='POST')
    try:
        response = (transport or post_json)(request, 30)
    except HTTPError as exc:
        reason = {401: 'Authentication failed', 403: 'Access denied', 429: 'Provider rate or quota limit reached'}.get(exc.code, 'Connection test failed')
        raise LanguageError(f'{reason} (HTTP {exc.code}, {route}). Model listing and inference can have different permissions. Check the applied key, model and institutional endpoint access. No automatic retry was sent.') from None
    except (URLError, TimeoutError, OSError):
        raise LanguageError('The inference endpoint could not be reached within 30 seconds. No automatic retry was sent.') from None
    except (ValueError, TypeError):
        raise LanguageError('The inference endpoint returned an invalid response. No experimental data was sent.') from None
    if not isinstance(response, dict) or response.get('error'):
        raise LanguageError('The provider returned an error or unsupported response. No experimental data was sent.')
    if config.api_format == 'responses':
        text_found = isinstance(response.get('output_text'), str) and bool(response['output_text'].strip())
        output = response.get('output', [])
        for item in output if isinstance(output, list) else []:
            if not isinstance(item, dict) or item.get('type') != 'message':
                continue
            content = item.get('content', [])
            for part in content if isinstance(content, list) else []:
                if (isinstance(part, dict) and part.get('type') == 'output_text'
                        and isinstance(part.get('text'), str) and part['text'].strip()):
                    text_found = True
        complete = response.get('status') in ('completed', None) and text_found
    else:
        choices = response.get('choices')
        complete = (isinstance(choices, list) and bool(choices) and isinstance(choices[0], dict)
                    and isinstance(choices[0].get('message'), dict) and bool(choices[0]['message'].get('content')))
    return {'route': route, 'request_accepted': True, 'complete_text_reply': bool(complete)}
