# dvo_python: Dense visual odometry in Python(3.6(.6)) 
> Coded up in a night! :)

Someone tweeted about [this elegant implementation ](https://github.com/muskie82/simple_dvo), and that's what made my day (rather, my night). I was like, "Hmm, a good refresher on dense SLAM would be to implement this, let me do it in Python."

## Big picture checklist

- [x] Barebones version of repo up
- [x] Chalk out repo structure, order in which to code
- [x] Put the order up as a checklist
- [ ] Code & Debug (on a pair of images)!
- [ ] Code & Debug on a sequence of images
- [ ] Get it to run on TUM RGB-D!
- [ ] Benchmark (time and accuracy)
- [ ] Finish up documentation and README
- [ ] Take a moment to revel in a sense of accomplishment ;)
- [ ] Get some sleep!!! :)


## Micromanagement

- [x] Work out dependencies (`numpy`, `OpenCV`, some SE(3) package(??), `matplotlib`)
- [x] Read in a pair of pointclouds and visualize them
- [x] Construct image and depth pyramids
- [x] Compute the residual (warping error)
- [x] Implement SE(3) routines
- [x] Implement image gradient computation
- [ ] Compute Jacobian of the error function
- [ ] Write a Gauss-Newton optimizer
- [ ] Robustify the error function (IRLS / M-Estimators)
- [ ] Debug the two-image alignment case
- [ ] Extend to a sequence of several images
- [ ] Setup class to load a TUM RGB-D sequence and run `dvo _python` on it.
- [ ] Debug!
- [ ] Check for any possibile visualization glitches/enhancements.


## Activity Log

Times are in 24-hour format.

* 2000 - 2020: Chalk out action plan.
* 2020 - 2050: Dinner break.
* 2100 - 2130: Download a sequence from TUM RGB-D, load and display stuff
* 2140 - 2300: Build pyramid levels (plus a lot of interruptions :|)
* 2330 - 0115: Compute the photometric warp error
* 0115 - 0145: Get SE(3) helper functions in
* 0200 - 0220: Image gradient computation, fetch SE(3) Jacobian helper functions
