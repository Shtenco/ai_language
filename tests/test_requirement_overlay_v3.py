from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from sinergy_requirement_binding.binding import BindingInvariantError, bind_change_set_requirements
from sinergy_requirement_binding.overlay_v2 import materialize_requirement_overlay_v2
from sinergy_requirement_binding.overlay_v3 import (
    materialize_requirement_overlay_v3,
    requirement_overlay_v3_digest,
    validate_requirement_overlay_v3,
)

ROOT=Path(__file__).resolve().parents[1]
OVERLAY_V2=ROOT/'requirements/causal_accounting_requirements_v2.json'
OVERLAY_V3=ROOT/'requirements/accounting_close_requirements_v3.json'
SCHEMA_V3=ROOT/'schemas/requirement_overlay_v3.schema.json'

def load(path): return json.loads(path.read_text(encoding='utf-8'))

def base_registry():
    return {
        'schema':'sinergy.requirement-registry/v1',
        'as_of':'2026-08-17',
        'default_verification_status':'PENDING_RUNNER',
        'requirements':[
            {'id':'REQ-FIN-001','statement':'Every canonical reporting entry must conserve debits and credits exactly.','owner':'Shtenco/synergy_financial_os','implementation':['sinergy_financial_os/reporting.py'],'tests':['tests/test_reporting.py'],'implementation_status':'ADDRESSED','verification_status':'PENDING_RUNNER'}
        ],
    }

def materialized_v2(): return materialize_requirement_overlay_v2(base_registry(),load(OVERLAY_V2))

def test_overlay_v3_schema_and_semantics():
    overlay=load(OVERLAY_V3);schema=load(SCHEMA_V3)
    jsonschema.Draft202012Validator.check_schema(schema);jsonschema.Draft202012Validator(schema).validate(overlay)
    normalized=validate_requirement_overlay_v3(overlay)
    assert normalized['base_overlay_schema']=='sinergy.requirement-overlay/v2'
    assert len(normalized['requirements'])==7
    assert len(requirement_overlay_v3_digest(overlay))==64

def test_v3_materializes_only_on_exact_v2_materialized_registry():
    v2=materialized_v2();v3=materialize_requirement_overlay_v3(v2.registry,load(OVERLAY_V3),expected_base_materialized_registry_digest=v2.registry_digest)
    assert len(v3.registry['requirements'])==16
    assert 'REQ-FIN-015' in v3.added_requirement_ids
    assert 'REQ-CHAIN-003' in v3.added_requirement_ids
    assert 'REQ-PORTAL-003' in v3.added_requirement_ids
    bound=bind_change_set_requirements({'schema':'engineering.change-set/v1','requirement_refs':['REQ-FIN-015','REQ-CHAIN-003']},v3.registry,input_requirement_registry_digest=v3.registry_digest)
    assert bound.status=='BOUND';assert bound.blockers==()

def test_v3_rejects_stale_base_materialized_digest():
    v2=materialized_v2()
    with pytest.raises(BindingInvariantError,match='base materialized registry digest mismatch'):
        materialize_requirement_overlay_v3(v2.registry,load(OVERLAY_V3),expected_base_materialized_registry_digest='00'*32)

def test_v3_cannot_replace_v2_requirement_id():
    v2=materialized_v2();overlay=load(OVERLAY_V3);overlay['requirements'][0]['id']='REQ-FIN-011'
    with pytest.raises(BindingInvariantError,match='cannot replace existing IDs'):
        materialize_requirement_overlay_v3(v2.registry,overlay,expected_base_materialized_registry_digest=v2.registry_digest)
