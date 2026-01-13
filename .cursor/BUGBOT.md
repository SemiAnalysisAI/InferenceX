# Review Guidelines
1. Ignore any problems in utils/evals/*
2. If any code end up creating a directory in `/workspace`, it is a critical error and should have the title: "Potential EACCES Error". A file in `/workspace` is ok.
3. If any changed file contains /(?:^|\s)(TODO|FIXME)(?:\s*:|\s+)/, then:
    - Add a non-blocking Bug titled "TODO/FIXME comment found"
    - Body: "Replace TODO/FIXME with a tracked issue reference, e.g., `TODO(#1234): ...`, or remove it."
    - If the TODO already references an issue pattern /#\d+|[A-Z]+-\d+/, mark the Bug as resolved automatically.