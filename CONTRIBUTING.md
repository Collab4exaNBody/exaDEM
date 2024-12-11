Add or Change a feature
=======================



Style
=====

This project is formatted with clang-format using the style file at the root of the repository. Please run clang-format before sending a pull request.

Commit messages should be in the imperative mood, as described in the Git contributing file:

```
    Describe your changes in imperative mood, e.g. "make xyzzy do frotz" instead of "[This patch] makes xyzzy do frotz" or "[I] changed xyzzy to do frotz", as if you are giving orders to the codebase to change its behaviour.
```

Tests
=====

Please verify the tests pass. Use the following commands in your build directory:

```
ctest --test-dir example
```

If you are adding functionality, add tests accordingly.

Pull request process
====================

Every pull request undergoes a code review. During the code review, if you make changes, add new commits to the pull request for each change. Once the code review is complete, rebase against the master branch and squash into a single commit.

Given the size of the code, only one reviewer will be assigned to you. If the assignment time exceeds one week, please tag `rprat-pro` or `carrardt` in the conversion.
