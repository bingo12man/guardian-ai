
package guardian.g3

import rego.v1

default allow := false

# Only trusted, enabled identities using
# assigned tools may perform operations.
identity_ok if {
    input.principal.exists == true
    input.principal.enabled == true
    input.tool_authorized == true
}

# Reading a document requires both existence
# and sufficient clearance.
allow if {
    identity_ok
    input.action.tool == "read_document"
    input.document.exists == true
    input.principal.clearance_level >= input.document.level
}

# Local summary creation requires the
# necessary prior reads and unused summary ID.
allow if {
    identity_ok
    input.action.tool == "create_summary"
    input.workflow.sources_read == true
    input.workflow.summary_id_available == true
}

# Publishing requires a pre-existing
# public-derived summary.
allow if {
    identity_ok
    input.action.tool == "publish_summary"
    input.summary.exists == true
    input.summary.classification == "public"
}

reason_code := "ALLOW" if {
    allow
} else := "DENIED_BY_POLICY"

decision := {
    "allow": allow,
    "reason_code": reason_code,
    "policy_version": "g3-v1",
}
