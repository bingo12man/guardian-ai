
package guardian.g3

import rego.v1

default allow := false

identity_ok if {
    input.principal.exists == true
    input.principal.enabled == true
    input.tool_authorized == true
}

allow if {
    identity_ok
    input.action.tool == "read_document"
    input.document.exists == true
    input.principal.clearance_level >= input.document.level
}

allow if {
    identity_ok
    input.action.tool == "create_summary"
    input.workflow.sources_read == true
    input.workflow.summary_id_available == true
}

# Changed rule:
# Public classification AND trusted approval.
allow if {
    identity_ok
    input.action.tool == "publish_summary"
    input.summary.exists == true
    input.summary.classification == "public"
    input.approval.approved == true
}

reason_code := "ALLOW" if {
    allow
} else := "DENIED_BY_POLICY"

decision := {
    "allow": allow,
    "reason_code": reason_code,
    "policy_version": "g3-v2",
}
