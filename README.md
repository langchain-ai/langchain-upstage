# 🦜️🔗 LangChain Upstage

This repository contains 1 package with Upstage integrations with LangChain:

- [langchain-upstage](https://pypi.org/project/langchain-upstage/) integrates [Upstage](https://www.upstage.ai/).


## Release process

In order to release a new version of `langchain-upstage`, follow these steps.

1. Update the version of `langchain-upstage` in `libs/upstage/pyproject.toml` file to the next release candidate version. For example, if the latest version is `0.1.5`, next rc version should be `0.1.6rc0`.
2. Run `release` action to release the rc version.
3. Verify your changes using released rc version.
4. If the rc version has a bug, fix the bug and repeat the step 1~3 to release another rc version. (`0.1.6rc1` in this case)
5. If the rc version works as expected, remove the rc from the version (`0.1.6` in this case) and release it again.

