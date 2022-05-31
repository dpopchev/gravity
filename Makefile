# set preferred shell
SHELL := /usr/bin/env bash
PYPATHLIB := PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}./lib:../nrpytutorial"
INTPR := $(PYPATHLIB) python3-latest

## -----------------------------------------------------------------------------
## Makefile to rule a python project. It is written in GNU Make and expects bash.
## Use with: make <target>

.PHONY: help ## show this help menu
help:
	@sed -nr '/#{2}/{s/\.PHONY: /-- /; s/#{2} /: /; p;}' ${MAKEFILE_LIST}

.DEFAULT_GOAL: help

SRC_DIR := src
LIB_DIR := lib
TESTS_DIR := tests
DUMMIES_DIR := .dummies
DOC_DIR := doc
DIRS := $(SRC_DIR) $(LIB_DIR) $(TESTS_DIR) $(DUMMIES_DIR) $(DOC_DIR)

MKDIR := mkdir --parents

$(DIRS):
	@$(MKDIR) $@
	@grep --quiet --fixed-strings "$(DUMMIES_DIR)" .gitignore || echo "$(DUMMIES_DIR)" >> .gitignore

FIND_EXCLUDE := -path '*/.pytest_cache/*'
FIND_EXCLUDE += -o -path '*/__pycache__/*'
FIND_EXCLUDE += -o -path '*/fixtures/*'
FIND_EXCLUDE += -o -type f -not -name '__init__.py'
FIND_PYFILES = $(shell find $(1)/ $(FIND_EXCLUDE) -print 2> /dev/null)

SRCS := $(call FIND_PYFILES,$(SRC_DIR))
LIBS := $(call FIND_PYFILES,$(LIB_DIR))
TESTS := $(call FIND_PYFILES,$(TESTS_DIR))

PYFILES = $(LIBS) $(SRCS)
LINTFILES = $(PYFILES) $(TESTS)
ifdef FILE
	PYFILES = $(FILE)
	LINTFILES = $(FILE)
endif

UNITTEST_PYFILES := $(call FIND_PYFILES,$(TESTS_DIR))

COMPILE_DUMMIES_DIR := $(DUMMIES_DIR)/compile
PYFILES_MOD_DUMMIES_COMPILE := $(addprefix $(COMPILE_DUMMIES_DIR)/,$(PYFILES:%=%.mod))

LINT_DUMMIES_DIR := $(DUMMIES_DIR)/lint
PYFILES_MOD_DUMMIES_LINT := $(addprefix $(LINT_DUMMIES_DIR)/,$(LINTFILES:%=%.mod))

FORMAT_DUMMIES_DIR := $(DUMMIES_DIR)/format
PYFILES_MOD_DUMMIES_FORMAT := $(addprefix $(FORMAT_DUMMIES_DIR)/,$(LINTFILES:%=%.mod))

UNITTEST_DUMMIES_DIR := $(DUMMIES_DIR)/unittest
PYFILES_MOD_DUMMIES_UNITTESTS := $(addprefix $(UNITTEST_DUMMIES_DIR)/,$(PYFILES:%=%.mod))

DOCTEST_DUMMIES_DIR := $(DUMMIES_DIR)/doctest
# TODO pytest fails to run doctest of files having .py suffix
# TODO follow https://github.com/pytest-dev/pytest/issues/4476
# TODO follow https://github.com/pytest-dev/pytest/issues/3520
# TODO in current implementation only place for bare python files is in SRC
# TODO thus we are skipping them until better resolution is found
PYFILES_MOD_DUMMIES_DOCTESTS := $(addprefix $(DOCTEST_DUMMIES_DIR)/,$(patsubst %,%.mod,$(filter %.py,$(PYFILES))))

BB_JUNIT_DIR = test-results/$(1)
BB_JUNIT_XML = $(shell echo "$(1)" | sed -rn 's/\//\./gp').xml

UML_DIR := $(DOC_DIR)/uml
UML_SUFFIX := uml
UML_FILES := $(wildcard $(UML_DIR)/*.$(UML_SUFFIX))
UML_DUMMIES_DIR := $(DUMMIES_DIR)/uml
UML_MOD_DUMMIES := $(addprefix $(UML_DUMMIES_DIR)/,$(UML_FILES:%=%.mod))

PRINT_INFO = printf -- "--- $(1) $(2)\n"

.PHONY: compile ## syntax check for specific FILE or all modified ones
COMPILE = $(INTPR) -m pyflakes $(1)
$(DUMMIES_DIR)/compile/%.mod: % | $(DUMMIES_DIR)
	@$(MKDIR) $(dir $@)
	@$(call PRINT_INFO,compile,$<)
	@$(call COMPILE,$<)
	@touch $@

compile: $(PYFILES_MOD_DUMMIES_COMPILE)

.PHONY: clean_compile
clean_compile:
	@rm --recursive --force $(DUMMIES_DIR)/compile

.PHONY: lint ## pep8 compatibility check for specific FILE or all modified ones
LINT = $(INTPR) -m pylint --fail-under=7.5 $(1)
ifdef IS_BB_PIPELINE
	LINT = $(INTPR) -m pylint --fail-under=7.5 --output-format=pylint_junit.JUnitReporter $(1) > $(call BB_JUNIT_DIR,lint)/$(call BB_JUNIT_XML,$(1))
endif
$(DUMMIES_DIR)/lint/%.mod: % | $(DUMMIES_DIR)
	@$(MKDIR) $(dir $@)
	@$(call PRINT_INFO,lint,$<)
	@$(call LINT,$<)
	@touch $@

lint: $(PYFILES_MOD_DUMMIES_LINT)

.PHONY: clean_lint
clean_lint:
	@rm --recursive --force $(DUMMIES_DIR)/lint

.PHONY: masteruml ## pyreverse to generate master uml diagram
masteruml:
	@touch $(LIB_DIR)/__init__.py $(SRC_DIR)/__init__.py
	@$(PYPATHLIB) pyreverse $(LIB_DIR)/ $(SRC_DIR)/ -o png -p $@
	@rm -f $(LIB_DIR)/__init__.py $(SRC_DIR)/__init__.py

.PHONY: format ## attempt to auto fix pep8 violations on a FILE or all modified ones
FORMAT = $(PYPATHLIB) autopep8 --in-place --aggressive --aggressive --hang-closing $(1)
$(DUMMIES_DIR)/format/%.mod: % | $(DUMMIES_DIR)
	@$(MKDIR) $(dir $@)
	@$(call PRINT_INFO,format,$<)
	@$(call FORMAT,$<)
	@touch $@

format: $(PYFILES_MOD_DUMMIES_FORMAT)

.PHONY: clean_format
clean_format:
	@rm --recursive --force $(DUMMIES_DIR)/format

MAIN := $(SRC_DIR)/mainexec
.PHONY: run ## execute either main executable or pass file path with FILE=path/to/file
RUN = $(INTPR) $(1) $(2)
run:
ifndef FILE
	@$(call RUN,$(MAIN),$(ARGS))
else
	@$(call RUN,$(FILE),$(ARGS))
endif

.PHONY: debug ## as run, but with debugger
DEBUG = $(INTPR) -m pdb $(1) $(2)
debug:
ifndef FILE
	@$(call DEBUG,$(MAIN),$(ARGS))
else
	@$(call DEBUG,$(FILE),$(ARGS))
endif

.PHONY: profile ## create set of statistics of code execution
PROFILE_FORMAT := pstats
PROFILE_STATS := profile.$(PROFILE_FORMAT)
CREATE_PROFILE = $(INTPR) -m cProfile -o $(1) $(2) $(3)
$(PROFILE_STATS):
ifndef FILE
	@$(call CREATE_PROFILE,$@,$(MAIN),$(ARGS))
else
	@$(call CREATE_PROFILE,$@,$(FILE),$(ARGS))
endif
profile: $(PROFILE_STATS)

.PHONY: gprof2dot ## represents profile graph as static image
VIEW_FORMAT := png
VIEW_OUT := $(PROFILE_STATS).$(VIEW_FORMAT)

$(VIEW_OUT): $(PROFILE_STATS)
	@gprof2dot --format=$(PROFILE_FORMAT) $< | dot -T$(VIEW_FORMAT) -o $@

VIEWER := okular
gprof2dot: $(VIEW_OUT)
	@$(VIEWER) $<

.PHONY: snakeviz ## creates profile graph as dynamical web page
BROWSER := firefox
snakeviz: $(PROFILE_STATS)
	@snakeviz --browser $(BROWSER) $(PROFILE_STATS)

EXTRACT_TESTS_PARENT_DIR = $(shell echo $(basename $(1)) | sed -rn 's/$(LIB_DIR)|$(SRC_DIR)/$(TESTS_DIR)/p')
PYTEST := $(INTPR) -m pytest -v
PYTEST_ALLOW_EMPTY_TESTS := || [ $$? -eq 5 ] && exit 0

.PHONY: unittest ## run unitttests on modified files
UNITTEST = $(PYTEST) "$(1)" $(PYTEST_ALLOW_EMPTY_TESTS)
ifdef IS_BB_PIPELINE
	UNITTEST = $(PYTEST) --junit-xml=$(call BB_JUNIT_DIR,unittest)/$(call BB_JUNIT_XML,$(1)) "$(1)" $(PYTEST_ALLOW_EMPTY_TESTS)
endif

$(UNITTEST_DUMMIES_DIR)/%.mod: % | $(DUMMIES_DIR)
	@$(MKDIR) $(dir $@)
	@$(MKDIR) $(call EXTRACT_TESTS_PARENT_DIR,$<)
	@$(call PRINT_INFO,unittest,$<)
	@$(call UNITTEST,$(call EXTRACT_TESTS_PARENT_DIR,$<))
	@touch $@

unittest: $(PYFILES_MOD_DUMMIES_UNITTESTS)

.PHONY: clean_unittest
clean_unittest:
	@rm --recursive --force $(DUMMIES_DIR)/unittest

.PHONY: doctest ## run doctests on modified files
DOCTEST = $(PYTEST) --doctest-modules "$(1)" $(PYTEST_ALLOW_EMPTY_TESTS)
ifdef IS_BB_PIPELINE
	DOCTEST = $(PYTEST) --doctest-modules --junit-xml=$(call BB_JUNIT_DIR,doctest)/$(call BB_JUNIT_XML,$(1)) "$(1)" $(PYTEST_ALLOW_EMPTY_TESTS)
endif

$(DOCTEST_DUMMIES_DIR)/%.mod: % | $(DUMMIES_DIR)
	@$(MKDIR) $(dir $@)
	@$(call PRINT_INFO,doctest,$<)
	@$(call DOCTEST,$<)
	@touch $@

doctest: $(PYFILES_MOD_DUMMIES_DOCTESTS)

.PHONY: clean_doctest
clean_doctest:
	@rm --recursive --force $(DUMMIES_DIR)/doctest

COVERAGE_PATHS = --cov=src --cov=lib
COVERAGE_TESTS = $(TESTS_DIR)
COVERAGE_PARAMS := --cov-branch --cov-fail-under=75 --cov-report=term
ifdef FILE
	COVERAGE_PATHS = --cov=$(dir $(FILE))
	COVERAGE_TESTS = $(call EXTRACT_TESTS_PARENT_DIR,$(FILE))
	COVERAGE_PARAMS := $(subst term,html,$(COVERAGE_PARAMS))
endif

COVERAGE = $(PYTEST) $(COVERAGE_PATHS) $(COVERAGE_PARAMS) $(COVERAGE_TESTS)
.PHONY: coverage ## evaluate test coverage
coverage: clean_coverage
	@$(COVERAGE)

.PHONY: clean_coverage
clean_coverage:
	@rm --force --recursive .coverage htmlcov/

.PHONY: test ## execute both docttests and unittests
test: doctest unittest coverage

.PHONY: clean_test
clean_test: clean_doctest clean_unittest clean_coverage

PIP := $(INTPR) -m pip
PIP_INSTALL = $(PIP) install $1 > /dev/null
PIP_SHOW = $(PIP) show $1

REQ_FILE := requirements.txt
REPLACE_REQ := grep --perl-regexp --only-matching '^[\w-]+' $(REQ_FILE)
REPLACE_REQ += | sort --unique
REPLACE_REQ += | xargs -n1 -I{} sh -c '$(call PIP_SHOW,{})'
REPLACE_REQ += | sed -rn '/Name:/{s/Name: (.*)/\1==/;H;}; /Version:/{s/Version: (.*)/\1/;H;}; $${x;s/\n//;s/=\n/=/g;p;}'

.PHONY: requirement ## pip wrapper to add requirement, pass with PACKAGE=<name>
requirement:
	@$(call PIP_INSTALL,$(PACKAGE))
	@echo $(PACKAGE) >> $(REQ_FILE)
	@$(REPLACE_REQ) > $(REQ_FILE)_new
	@mv --backup=numbered $(REQ_FILE)_new $(REQ_FILE)

.PHONY: requirements ## install packages from local requirements.txt
requirements:
	@$(call PIP_INSTALL,-r requirements.txt)

.PHONY: build
build:
	$(error empty rule)

.PHONY: clean ## reset modification tracking to run test on all files
clean: clean_compile clean_lint clean_format clean_test
	@rm --recursive --force $(DUMMIES_DIR)

.PHONY: uml ## create uml diagrams
UML = plantuml $(1)
$(DUMMIES_DIR)/uml/%.mod: % | $(DUMMIES_DIR)
	@$(MKDIR) $(dir $@)
	@$(call PRINT_INFO,uml,$<)
	@$(call UML,$<)
	@touch $@

uml: $(UML_MOD_DUMMIES)

TEMPLATE_MARKER := \#~
TEMPLATE_START = $(TEMPLATE_MARKER) $(1) \{{3}
TEMPLATE_END := $(TEMPLATE_MARKER) \}{3}
CREATE_TEMPLATE = sed -rn '/$(call TEMPLATE_START,$(1))/,/$(TEMPLATE_END)/{ /$(call TEMPLATE_START,$(1))/b; /$(TEMPLATE_END)/b; s/$(TEMPLATE_MARKER)//p}' $(MAKEFILE_LIST) > $(2)

TEMPLATE_FILES := $(SRC_DIR)/mainexec $(LIB_DIR)/supportive.py $(TESTS_DIR)/mainexec/test_main.py $(TESTS_DIR)/supportive/test_say.py ./.gitignore
TEMPLATE_TARGETS := $(notdir $(TEMPLATE_FILES))
.PHONY: $(TEMPLATE_TARGETS)
$(TEMPLATE_TARGETS):
	@$(MKDIR) $(dir $(filter %/$@,$(TEMPLATE_FILES))) || true
	@$(call CREATE_TEMPLATE,$@,$(filter %/$@,$(TEMPLATE_FILES)))

GIT := .git

$(GIT):
	@git init
	@git add .
	@git commit --allow-empty -m 'init python project'

.PHONY: sample ## build a sample python project
sample: $(TEMPLATE_TARGETS) | $(GIT)
	@git status

#~ mainexec {{{
#~#!/usr/bin/env python3
#~'''Simple script to demonstrate project structure and Makefile abilities'''
#~
#~from supportive import SupportiveClass, supportive_function
#~
#~
#~def main():
#~    '''main executable, C-style like, but not mandatory'''
#~    print('Hello from mainexec, a stand alone python script')
#~
#~    print('It is taking advantage of supportive module, found under lib/')
#~
#~    support = SupportiveClass('here is my number')
#~    support_said = support.say()
#~    print(support_said)
#~
#~    support_said = supportive_function('so call me maybe')
#~    print(support_said)
#~
#~
#~if __name__ == '__main__':
#~    main()
#~ }}}
#~
#~ supportive.py {{{
#~'''Module with supportive functionality'''
#~
#~# note, comments start with hashtag
#~# some linter check can be locally disabled
#~
#~
#~class SupportiveClass: # pylint: disable=too-few-public-methods
#~    '''Supportive class, can be very supportive
#~    Sample usage:
#~    >>> support = SupportiveClass('you can do it')
#~    >>> support.say()
#~    'Supportive class says: you can do it'
#~    '''
#~
#~    def __init__(self, message):
#~        self.message = message
#~
#~    def say(self):
#~        '''share some supportive toughts'''
#~        return f'Supportive class says: {self.message}'
#~
#~
#~def supportive_function(message):
#~    '''functions can also provide supportive messages'''
#~    return f'Supportive function says: {message}'
#~ }}}
#~
#~ test_main.py {{{
#~'''unittest for main script function'''
#~import unittest
#~from unittest.mock import patch
#~import io
#~
#~import importlib.machinery
#~import importlib.util
#~from pathlib import Path
#~
#~
#~class TestMainFunctionStdoutPrint(unittest.TestCase):
#~    '''main function stdout verification'''
#~
#~    def setUp(self):
#~        cwd = Path(__file__).parent
#~        mainexec_path = str(cwd.joinpath('..', '..', 'src', 'mainexec'))
#~
#~        loader = importlib.machinery.SourceFileLoader(
#~            'mainexec', mainexec_path)
#~        spec = importlib.util.spec_from_loader('mainexec', loader)
#~        mainexec = importlib.util.module_from_spec(spec)
#~        loader.exec_module(mainexec)
#~
#~        self.mainexec = mainexec
#~
#~    def test_main_function_stdout(self):
#~        '''basic std out inf printed by main function'''
#~
#~        with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
#~
#~            self.mainexec.main()
#~            got = fake_stdout.getvalue()
#~
#~            expt = (
#~                "Hello from mainexec, a stand alone python script\n"
#~                "It is taking advantage of supportive module, found under lib/\n"
#~                "Supportive class says: here is my number\n"
#~                "Supportive function says: so call me maybe\n")
#~
#~            self.assertMultiLineEqual(
#~                got, expt, 'Main function prints info to stdout')
#~
#~
#~if __name__ == '__main__':
#~    unittest.main()
#~ }}}
#~
#~ test_say.py {{{
#~'''unittest for supportive module'''
#~import unittest
#~
#~from supportive import SupportiveClass, supportive_function
#~
#~
#~class TestSupportiveClass(unittest.TestCase):
#~    '''test supportive class functionality'''
#~
#~    def test_say_method(self):
#~        '''test say method'''
#~
#~        what_to_say = 'pen is mightier than the sword'
#~
#~        support = SupportiveClass(what_to_say)
#~        what_was_said = support.say()
#~
#~        expt = f'Supportive class says: {what_to_say}'
#~        self.assertEqual(what_was_said, expt, 'say method speaks the truth')
#~
#~
#~class TestSupportiveFunctions(unittest.TestCase):
#~    '''test supportive_function'''
#~
#~    def test_support_message(self):
#~        '''test if support message is returned'''
#~
#~        what_to_say = 'functions are mightier than classes'
#~
#~        what_was_said = supportive_function(what_to_say)
#~
#~        expt = f'Supportive function says: {what_to_say}'
#~        self.assertEqual(what_was_said, expt, 'supportive function supported')
#~
#~
#~if __name__ == '__main__':
#~    unittest.main()
#~ }}}
#~
#~ .gitignore {{{
#~*.pyc
#~__pycache__/
#~.dummies/
#~venv/
#~env/
#~*~
#~ }}}
#~
#~ bitbucket-pipelines.yml {{{
#~image: python:x.x.x
#~
#~pipelines:
#~  default:
#~    - parallel:
#~        - step:
#~            name: Lint
#~            script:
#~              - make requirements
#~              - mkdir --parents test-results/lint
#~              - export IS_BB_PIPELINE=1
#~              - make -k lint
#~        - step:
#~            name: Doctest
#~            script:
#~              - make requirements
#~              - mkdir --parents test-results/doctest
#~              - export IS_BB_PIPELINE=1
#~              - make -k doctest
#~        - step:
#~            name: Unittest
#~            script:
#~              - make requirements
#~              - mkdir --parents test-results/unittest
#~              - export IS_BB_PIPELINE=1
#~              - make -k unittest
#~        - step:
#~            name: Coverage
#~            script:
#~              - make requirements
#~              - mkdir --parents test-results/unittest
#~              - export IS_BB_PIPELINE=1
#~              - make -k coverage
#~ }}}
