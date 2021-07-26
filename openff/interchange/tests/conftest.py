def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if "not slow" not in terminalreporter.config.option.markexpr:
        if "skipped" in terminalreporter.stats:
            num_skipped_tests = len(terminalreporter.stats["skipped"])
            # TODO: Detect if this run is minimal_tests.yaml and do not error
            # based on number of skipped tests
            if num_skipped_tests > 120:
                raise Exception(f"Too many skipped tests: {num_skipped_tests}")
