# 🦜️🔗 LangChain Upstage

This repository contains 1 package with Upstage integrations with LangChain:

- [langchain-upstage](https://pypi.org/project/langchain-upstage/) integrates [Upstage](https://www.upstage.ai/).

## Initial Repo Checklist (Remove this section after completing)

Workflow code

- [ ] Add secrets as env vars in .github/workflows/_release.yml
- [ ] Populate .github/workflows/_release.yml with `on.workflow_dispatch.inputs.working-directory.default`
- [ ] Configure `LIB_DIRS` in .github/scripts/check_diff.py

In github

- [ ] Add integration testing secrets in Github (ask Erick for help)
- [ ] Add partner collaborators in Github (ask Erick for help)
- [ ] "Allow auto-merge" in General Settings 
- [ ] Only "Allow squash merging" in General Settings
- [ ] Set up ruleset matching CI build (ask Erick for help)
    - name: ci build
    - enforcement: active
    - bypass: write
    - target: default branch
    - rules: restrict deletions, require status checks ("CI Success"), block force pushes

Pypi

- [ ] Add new repo to test-pypi and pypi trusted publishing (ask Erick for help)

Slack

- [ ] Set up release alerting in Slack (ask Erick for help)
