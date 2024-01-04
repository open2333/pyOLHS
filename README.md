# pyOLHS
--------------------------------------------------------------------------------

pyOLHS is a Python package that provides Optimal LatinHyperCube sampling.

The introduction of Algorithms is on my WeChat Official Account.

The Python version of the OLHS algorithm is less efficient. In the future, I will release an open-source C++ version optimized with openMP and TBB. 

Additionally, I plan to update other multi-disciplinary optimization algorithms, such as Kriging models and other optimization algorithms like Bayesian Optimization.

## Getting Started
### discrete variable
```python
bound = [[0, 1.0, 0.01], [0, 1.0, 0.01]]
#2024.1.4:initseed means to fix the origin lathin sampling result optseed means to fix the optimization method
#if you want to get the same optimization method , fix them both, 
a = OLHS(bound, 20, 100,initseed=1,optseed=1) 
#if you want to see different optimization result from the same origin latin result, only fix initseed
a = OLHS(bound, 20, 100,initseed=1)

c = a.sampling()
```
<img src="https://github.com/open2333/pyOLHS/assets/43056772/65a9ff70-8442-46c8-85bb-d8b2048bd433" width="500" height="400">

### continuous variable
```python
bound = [[0, 1.0,], [0, 1.0]]
a = OLHS(bound, 20, 1000,"center")
c = a.sampling()
```
<img src="https://github.com/open2333/pyOLHS/assets/43056772/39a1c425-0a24-4190-ab87-3042d31c4ceb" width="500" height="400">

## Notes:I convert my C++ code into this Python version, but for some reason, the performance is not as good as the C++ version. Below are the results of the C++ execution.


![olhs](https://github.com/open2333/pyOLHS/assets/43056772/9ab7db3a-7175-4cbe-8f8d-42d05183a304)


<img src="https://github.com/open2333/pyOLHS/assets/43056772/fdbbc989-6e4b-4c98-a5c3-8223109cd4ff" width="600" height="400">

## Release Note:
-2024.1.4 add random seed to both origin latin sampling and optimization algorithm


## Communication
* BiliBili: [allenalexyan](https://space.bilibili.com/319245648?spm_id_from=333.1296.0.0)
* Email:[yanjz111@gmail.com](mailto:yanjz111@gmail.com)
* 微信公众号
  <img src="https://github.com/open2333/pyOLHS/assets/43056772/bfe5eaed-8af3-4d3d-9473-a2fb763343f9" width="200" height="200">




## References

Jin R, Chen W, Sudjianto A. An efficient algorithm for constructing optimal design of computer experiments[C]//International design engineering technical conferences and computers and information in engineering conference. 2003, 37009: 545-554.

## License

PyOLHS has a BSD-style license, as found in the [LICENSE](LICENSE) file.
