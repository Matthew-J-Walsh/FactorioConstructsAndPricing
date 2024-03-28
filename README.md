# FactorioConstructsAndPricing
 
## Description
The purpose of this project is the implementation of price modeling in Factorio in order to find and analyze optimal factories in the game Factorio. This project is the implementation of a set of articles which can be accessed here: (https://drive.google.com/drive/folders/1G-ogarwaSEfp_JFCDxrdPlC4wEbExnSA?usp=sharing).

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)

## Installation
This project is exclusively written in python and all relevent imports are in (globalsandimports.py). Notable non-standard imports are:
 - highspy
 - pulp
 - pyscipopt
 - numexpr

## Usage
An example usage is provided in (vanillarun.py). Logging is mostly done with the logging module. The vanilla Factorio data.raw file is saved in (vanilla-rawdata.json). The general pipeline for using this tools is as follows:
1. Initialize an instance of FactorioInstance from a data.raw file.
2. Create a FactorioFactoryChain instance to hold the chain of created factories.
3. Create an initial pricing model as a CompressedVector of Fractions and an initial tech level using FactorioInstance.technological_limitation_from_specification.
4. Add factories with FactorioFactoryChain.add.
5. Save the factory setups and pricing models in excel for easy reading with FactorioFactoryChain.dump_to_excel.

## Documentation
No additional documentation is currently provided. Most if not all code is commented and has type annotations.

## Contributing
Currently this is a personal project. If you have suggestions or questions I'll try to respond in a timely manner.

## License
This project is licensed under the GNU General Public License v3.0.

You can find the full text of the license in the [LICENSE](LICENSE) file.

### GNU General Public License v3.0 (GPL-3.0)

The GNU General Public License is a free, copyleft license for software and other kinds of works. The license provides users with the freedom to run, study, share, and modify the software.

#### Summary of key terms:
- **Copyleft:** You are free to distribute copies and modified versions of the software, but you must make the source code available under the same license terms.
- **Distribution:** Any conveyance of the software, including distributing it to users over a network.
- **Modification:** Any change to the software's source code, including adding, removing, or altering functionality.

For detailed information and the full text of the license, please refer to the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html).

## Authors
 - Matthew Walsh (https://github.com/Matthew-J-Walsh)

## Acknowledgements
 - My parents for letting me work on this project while living in their house.
 - eigenchris on youtube for his amazing series that taught me Tensor Algebra and Tensor Calculus. (https://www.youtube.com/playlist?list=PLJHszsWbB6hrkmmq57lX8BV-o-YIOFsiG) (https://www.youtube.com/playlist?list=PLJHszsWbB6hpk5h8lSfBkVrpjsqvUGTCx)
 - All the legendary Mathmaticians that contributed to humanities understanding of Mathematical Optimization such as Harold W. Kuhn, Albert W. Tucker, William Karush, and Gyula Farkas.
 - All the contributors to the HiGHS, PuLP, and SCIP open source Linear Programming solvers.
