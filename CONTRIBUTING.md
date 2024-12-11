Contributing to ExaDEM
======================

Contributions to `exaDEM` are greatly appreciated.

Please take a moment to review this document in order to make the contribution process easy and effective for everyone involved.

Following these guidelines helps to communicate that you respect the time of the developers managing and developing this open source project. In return, they should reciprocate that respect in addressing your issue or assessing patches and features.

Using the issue tracker
=======================

The issue tracker is the preferred channel for bug reports, features requests and submitting pull requests, but please respect the following restrictions:

- Please do not use the issue tracker for personal support requests (contact directly the authors raphael.prat@cea.fr

Bug reports
===========

A bug is a demonstrable problem that is caused by the code in the repository. Good bug reports are extremely helpful - thank you!

Guidelines for bug reports:
---------------------------

- Use the GitHub issue search: check if the issue has already been reported.
- Check if the issue has been fixed: try to reproduce it using the latest master or development branch in the repository.
- Isolate the problem: ideally create a "[reduced test case].
- Tag your new issue with the label "bug".

A good bug report shouldn't leave others needing to chase you up for more information. Please try to be as detailed as possible in your report. What is your environment? What steps will reproduce the issue? What compiler(s) and OS experience the problem? What would you expect to be the outcome? All these details will help people to fix any potential bugs.


Pull requests
=============

Good pull requests - patches, improvements, new features - are a fantastic help. They should remain focused in scope and avoid containing unrelated commits.

Please ask first before embarking on any significant pull request (e.g. implementing features, refactoring code, porting to a different language), otherwise you risk spending a lot of time working on something that the project's developers might not want to merge into the project.

Please adhere to the coding conventions used throughout a project (indentation, accurate comments, etc.) and any other requirements (such as test coverage).

Feature requests are welcome. But take a moment to find out whether your idea fits with the scope and aims of the project. It's up to you to make a strong case to convince the project's developers of the merits of this feature. Please provide as much detail and context as possible.

Style
-----

This project is formatted with clang-format using the style file at the root of the repository. Please run clang-format before sending a pull request.

Commit messages should be in the imperative mood, as described in the Git contributing file:

```
    Describe your changes in imperative mood, e.g. "make xyzzy do frotz" instead of "[This patch] makes xyzzy do frotz" or "[I] changed xyzzy to do frotz", as if you are giving orders to the codebase to change its behaviour.
```

Tests
-----

Please verify the tests pass. Use the following commands in your build directory:

```
ctest --test-dir example
```

If you are adding functionality, add tests accordingly.

Pull request process
--------------------

Given the size of the code, only one reviewer will be assigned to you. If the assignment time exceeds one or two weeks, please tag `rprat-pro` or `carrardt` in the conversion.

IMPORTANT: By submitting a patch, you agree to allow the project owners to license your work under the the terms of the Apache2.0 License.
