from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from sinergy_requirement_binding.binding import (
    BindingInvariantError,
    bind_change_set_requirements,
    canonical_registry_digest,
)
from sinergy_requirement_binding.overlay_v2 import (
    materialize_requirement_overlay_v2,
    requirement_overlay_v2_digest,
    validate_requirement_overlay_v2,
)

ROOT=Path(__file__).resolve().parents[1]
OVERLAY_PATH=ROOT/'requirements/causal_accounting_requirements_v2.json'
SCHEMA_PATH=ROOT/'schemas/requirement_overlay_v2.schema.json'

def load(path): return json.loads(path.read_text(encoding='utf-8'))

def base_registry():
    return {
        'schema':'sinergy.requirement-registry/v1',
        'as_of':'2026-08-17',
        'default_verification_status':'PENDING_RUNNER',
        'requirements':[
            {
                'id':'REQ-FIN-001',
                'statement':'Every canonical reporting entry must conserve debits and credits exactly.',
                'owner':'Shtenco/synergy_financial_os',
                'implementation':['sinergy_financial_os/reporting.py'],
                'tests':['tests/test_reporting.py'],
                'implementation_status':'ADDRESSED',
                'verification_status':'PENDING_RUNNER',
            }
        ],
    }

def change_set(ref):
    return {'schema':'engineering.change-set/v1','requirement_refs':[ref]}

def test_overlay_schema_and_runtime_accept_causal_accounting_requirements():
    overlay=load(OVERLAY_PATH);schema=load(SCHEMA_PATH)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(overlay)
    normalized=validate_requirement_overlay_v2(overlay)
    assert len(normalized['requirements'])==8
    assert len(requirement_overlay_v2_digest(overlay))==64

def test_materialization_is_append_only_and_keeps_base_registry_v1_compatible():
    base=base_registry();overlay=load(OVERLAY_PATH)
    result=materialize_requirement_overlay_v2(base,overlay)
    assert result.base_registry_digest==canonical_registry_digest(base)
    assert result.registry['schema']=='sinergy.requirement-registry/v1'
    assert result.registry['requirements'][0]['id']=='REQ-FIN-001'
    assert result.added_requirement_ids[0]=='REQ-PAY-004'
    assert 'REQ-FIN-011' in result.added_requirement_ids
    assert len(result.registry['requirements'])==9

def test_overlay_cannot_replace_existing_requirement_id():
    base=base_registry();base['requirements'].append(load(OVERLAY_PATH)['requirements'][0])
    with pytest.raises(BindingInvariantError,match='cannot replace existing IDs'):
        materialize_requirement_overlay_v2(base,load(OVERLAY_PATH))

def test_new_change_set_is_bound_only_to_materialized_registry_digest():
    base=base_registry();materialized=materialize_requirement_overlay_v2(base,load(OVERLAY_PATH))
    bound=bind_change_set_requirements(change_set('REQ-FIN-011'),materialized.registry,input_requirement_registry_digest=materialized.registry_digest)
    assert bound.status=='BOUND';assert bound.blockers==()
    stale=bind_change_set_requirements(change_set('REQ-FIN-011'),materialized.registry,input_requirement_registry_digest=materialized.base_registry_digest)
    assert stale.status=='STALE_REGISTRY'
    assert stale.blockers==('REQUIREMENT_REGISTRY_DIGEST_MISMATCH',)
