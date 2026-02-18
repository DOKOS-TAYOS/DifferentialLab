### Third-Party Licenses for DifferentialLab

DifferentialLab (`differential-lab`) is distributed under the **MIT** license (see `license.md` in the project root).

This document lists the third-party libraries used and their licenses, to facilitate compliance
with their terms when redistributing DifferentialLab (as source code, installable package, binary,
or installer).

> Note: This list is based on the dependencies declared in `pyproject.toml` (and the matching
> `requirements.txt`). If you add or remove dependencies, update this file.

---

### 1. Runtime Dependencies

These libraries are used at application runtime.

| Library          | Version Range       | License Type                          |
|------------------|---------------------|---------------------------------------|
| **numpy**        | `>=2.0,<3.0`       | BSD-3-Clause                          |
| **matplotlib**   | `>=3.10,<4.0`      | Matplotlib License (BSD-style + PSF)  |
| **scipy**        | `>=1.15,<2.0`      | BSD-3-Clause                          |
| **python-dotenv**| `>=1.0,<2.0`       | BSD-3-Clause                          |
| **colorama**     | `>=0.4,<1.0`       | BSD-3-Clause                          |
| **PyYAML**       | `>=6.0,<7.0`       | MIT                                   |

---

### 2. Standard Library Components

The following standard-library modules are used and require no separate licensing:

| Module       | Purpose                     |
|--------------|-----------------------------|
| **tkinter**  | Desktop GUI (Tk/ttk)        |
| **logging**  | Application logging         |
| **csv**      | CSV file export             |
| **json**     | JSON file export            |
| **math**     | Mathematical functions      |
| **pathlib**  | File path management        |

---

### 3. Development and Tooling Dependencies (Optional)

These libraries are used only for development and are not distributed with the application.

| Library          | Version Range       | License Type |
|------------------|---------------------|--------------|
| **pytest**       | `>=8.0,<9.0`       | MIT          |
| **pytest-cov**   | `>=6.0,<7.0`       | MIT          |
| **ruff**         | `>=0.9,<1.0`       | MIT          |
| **mypy**         | `>=1.0,<2.0`       | MIT          |

---

### 4. Documentation Dependencies (Optional)

| Library                      | Version Range | License Type |
|------------------------------|---------------|--------------|
| **sphinx**                   | `>=8.0.0`     | BSD-2-Clause |
| **sphinx-rtd-theme**         | `>=3.0.0`     | MIT          |
| **myst-parser**              | `>=4.0.0`     | MIT          |
| **sphinx-autodoc-typehints** | `>=2.0.0`     | MIT          |

---

### 5. Full License Text: PyYAML (MIT)

PyYAML is used for loading predefined equation definitions in YAML format.

```
Copyright (c) 2017-2021 Ingy dot Net
Copyright (c) 2006-2016 Kirill Simonov

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

### 6. License Compliance Notes

- **MIT / BSD** licenses allow commercial use, modification, and redistribution.
  Main requirement: preserve copyright notice and license text in substantial copies.

- **Matplotlib License** (BSD-style + PSF): same permissive terms as BSD.

- **Recommended**: keep `license.md` and this `THIRD_PARTY_LICENSES.md` in all distributions.
