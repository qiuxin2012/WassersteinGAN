package com.intel.analytics.bigdl.wgan

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.transform.vision.image.augmentation._

object models {
  // dcgan Discriminator
  def dcganD(isize: Int, nz: Int, nc: Int, ndf: Int, nExtraLayers: Int = 0): Module[Float] = {
    val netD = Sequential()

    // input is nc x isize x isize
    netD.add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1, withBias = false)
      .setInitMethod(RandomNormal(0, 0.02))
      .setName(s"initial.conv.$nc-$ndf"))
    netD.add(LeakyReLU(0.2, true)
      .setName(s"initial.relu.$ndf"))

    var csize = isize / 2
    var cndf = ndf

    // Extra layers
    (0 until nExtraLayers).foreach{t =>
      netD.add(SpatialConvolution(cndf, cndf, 3, 3, 1, 1, 1, 1, withBias = false)
        .setInitMethod(RandomNormal(0, 0.02))
        .setName(s"extra-layers-$t.$cndf.conv"))
      netD.add(SpatialBatchNormalization(cndf)
        .setInitMethod(RandomNormal(1, 0.02), Zeros)
        .setName(s"extra-layers-$t.$cndf.batchnorm"))
      netD.add(LeakyReLU(0.2, true)
        .setName(s"extra-layers-$t.$cndf.relu"))
    }

    while (csize > 4) {
      val inFeat = cndf
      val outFeat = cndf * 2
      netD.add(SpatialConvolution(inFeat, outFeat, 4, 4, 2, 2, 1, 1, withBias = false)
        .setInitMethod(RandomNormal(0, 0.02))
        .setName(s"pyramid.$inFeat-$outFeat.conv"))
      netD.add(SpatialBatchNormalization(outFeat)
        .setInitMethod(RandomNormal(1, 0.02), Zeros)
        .setName(s"pyramid.$outFeat.batchnorm"))
      netD.add(LeakyReLU(0.2, true)
        .setName(s"pyramid.$outFeat.relu"))
      cndf = cndf * 2
      csize = csize / 2
    }

    // state size. K x 4 x 4
    netD.add(SpatialConvolution(cndf, 1, 4, 4, 1, 1, 0, 0, withBias = false)
      .setInitMethod(RandomNormal(0, 0.02))
      .setName(s"final.$cndf-1.conv"))

    netD.add(View(1).setNumInputDims(3))
    netD.add(Mean(1))
    netD
  }

  // dcgan generator
  def dcganG(isize: Int, nz: Int, nc: Int, ngf: Int, nExtraLayers: Int = 0): Module[Float] = {
    require(isize % 16 == 0, s"isize has to be a multiple of 16, but got $isize")
    var cngf = ngf / 2
    var tisize = 4
    while(tisize != isize) {
      cngf *= 2
      tisize *= 2
    }

    val netG = Sequential()
    // input is Z, going into a convolution
    netG.add(SpatialFullConvolution(nz, cngf, 4, 4, 1, 1, 0, 0, noBias = true)
      .setInitMethod(RandomNormal(0, 0.02))
      .setName(s"initial.$nz-$cngf.fullconv"))
    netG.add(SpatialBatchNormalization(cngf)
      .setInitMethod(RandomNormal(1, 0.02), Zeros)
      .setName(s"initial.$cngf.batchnorm"))
    netG.add(ReLU(true)
      .setName(s"initial.$cngf.relu"))

    var csize = 4
    var cndf = cngf

    while (csize < isize / 2) {
      netG.add(SpatialFullConvolution(cngf, cngf / 2, 4, 4, 2, 2, 1, 1, noBias = true)
        .setInitMethod(RandomNormal(0, 0.02))
        .setName(s"pyramid.$cngf-${cngf / 2}.fullconv"))
      netG.add(SpatialBatchNormalization(cngf / 2)
        .setInitMethod(RandomNormal(1, 0.02), Zeros)
        .setName(s"pyramid.${cngf / 2}.batchnorm"))
      netG.add(ReLU(true)
        .setName(s"pyramid.${cngf / 2}.relu"))
      cngf /= 2
      csize *= 2
    }

    // Extra layers
    (0 until nExtraLayers).foreach{t =>
      netG.add(SpatialFullConvolution(cngf, cngf, 3, 3, 1, 1, 1, 1, noBias = true)
        .setInitMethod(RandomNormal(0, 0.02))
        .setName(s"extra-layers-$t.${cngf}.fullconv"))
      netG.add(SpatialBatchNormalization(cngf)
        .setInitMethod(RandomNormal(1, 0.02), Zeros)
        .setName(s"extra-layers-$t.${cngf}.batchnorm"))
      netG.add(ReLU(true)
        .setName(s"extra-layers-$t.${cngf}.relu"))
    }
    netG.add(SpatialFullConvolution(cngf, nc, 4, 4, 2, 2, 1, 1, noBias = true)
      .setInitMethod(RandomNormal(0, 0.02))
      .setName(s"final.$cngf-$nc.fullconv"))
    netG.add(Tanh().setName(s"final.$nc.tanh"))

    netG
  }


}

object Mlp {
  def modelD(isize: Int, nz: Int, nc: Int, ndf: Int): Module[Float] = {
    val d = Sequential()
    d.add(View(nc * isize * isize))
      .add(Linear(nc * isize * isize, ndf))
      .add(ReLU(true))
      .add(Linear(ndf, ndf))
      .add(ReLU(true))
      .add(Linear(ndf, ndf))
      .add(ReLU(true))
      .add(Linear(ndf, 1))
      .add(Mean(1))
    d
  }

  def modelG(isize: Int, nz: Int, nc: Int, ngf: Int): Module[Float] = {
    val d = Sequential()
    d.add(View(nz))
      .add(Linear(nz, ngf))
      .add(ReLU(true))
      .add(Linear(ngf, ngf))
      .add(ReLU(true))
      .add(Linear(ngf, ngf))
      .add(ReLU(true))
      .add(Linear(ngf, nc * isize * isize))
      .add(Reshape(Array(nc, isize, isize)))
    d

  }
}
