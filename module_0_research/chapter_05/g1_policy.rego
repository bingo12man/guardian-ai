
package guardian.authz

import rego.v1

# Deny everything unless an explicit rule allows it.
default allow := false

# A trusted, enabled identity must use an authorized tool.
identity_ok if {
    input.principal.enabled == true
    input.tool_authorized == true
}

# Reading is permitted only if the document exists
# and clearance is high enough.
allow if {
    identity_ok
    input.action.tool == "read_document"
    input.document.exists == true
    input.principal.clearance_level >= input.document.level
}

# Creating a local summary requires verified
# workflow prerequisites from the gateway.
allow if {
    identity_ok
    input.action.tool == "create_summary"
    input.workflow.prerequisites_ok == true
}

# Public publication requires a public-derived summary.
allow if {
    identity_ok
    input.action.tool == "publish_summary"
    input.summary.exists == true
    input.summary.classification == "public"
}

# Human-readable decision information.
decision := {
    "allow": allow,
    "policy_version": "g1-v1",
    "reason": reason,
}

reason := "authorized" if {
    allow
} else := "denied_by_policy"
