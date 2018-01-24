package com.intel.analytics.bigdl.wgan

import com.intel.analytics.bigdl.dataset.{DataSet, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.mkl.MKL
import com.intel.analytics.bigdl.nn.Mean
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Adam, RMSprop}
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator}
import org.opencv.imgproc.Imgproc

object Train {

  def main(args: Array[String]): Unit = {
    val nCpu = args(0).toInt
    val imageSize = 64
    val nc = 3
    val nz = 100
    val ngf = 64
    val ndf = 64
    val batchSize = 64
    val numEpoch = 2000
    val dataroot = args(1) // "/home/xin/datasets/cifar-10/train"
//    val dataroot = "cifar10"

    Engine.init(1, nCpu, false)
    MKL.setNumThreads(nCpu)
    val localImageFrame = ImageFrame.read(dataroot)

    val imgPreprocess = Resize(imageSize, imageSize, Imgproc.INTER_CUBIC, false) ->
      ChannelNormalize(127.5f, 127.5f, 127.5f, 127.5f, 127.5f, 127.5f) ->
      MatToTensor() -> ImageFrameToSample()

    val cifar10 = imgPreprocess(localImageFrame).toLocal().array.filter(_.isValid)
      .map(_[Sample[Float]](ImageFeature.sample))
    val toBatch = SampleToMiniBatch(batchSize)

    val dataset = DataSet.array(cifar10) -> toBatch

    val netG = models.dcganG(imageSize, nz, nc, ngf)
    val netD = models.dcganD(imageSize, nz, nc, ndf)
//    val netG = Mlp.modelG(imageSize, nz, nc, ngf)
//    val netD = Mlp.modelD(imageSize, nz, nc, ndf)
    println(netG)
    println(netD)

    val (parametersG, gradParametersG) = netG.getParameters()
    val (parametersD, gradParametersD) = netD.getParameters()

    val fixedNoise = Tensor(batchSize, nz, 1, 1).apply1(_ => RandomGenerator.RNG.normal(0, 1).toFloat)
    netG.forward(fixedNoise)
    val noise = Tensor(batchSize, nz, 1, 1).apply1(_ => RandomGenerator.RNG.normal(0, 1).toFloat)
    val ones = Tensor(1).fill(64)
    val mones = Tensor(1).fill(-64)

//    val optimizerD = new Adam[Float](0.00005, beta1 = 0.5)
//    val optimizerG = new Adam[Float](0.00005, beta1 = 0.5)
    val optimizerD = new RMSprop[Float](0.00005)
    val optimizerG = new RMSprop[Float](0.00005)

    val dIterations = 5
    var gIters = 0
    var iteration = 0
    var epoch = 0
    val numSamples = dataset.toLocal().data(train = false).map(_.size()).reduce(_ + _)
    val numBatches = Math.ceil(numSamples.toFloat / batchSize)
    dataset.shuffle()
    var data = dataset.toLocal().data(true)
    val real = data.next().getInput().toTensor // .resize(3, 64, 64)
    Utils.save(real, "real.png")
    netD.training()
    netG.training()
    while (epoch < numEpoch) {
      val dIters = if (gIters < 25 || gIters % 500 == 0) {
        100
      } else {
        dIterations
      }

      var j = 0
      while (j < dIters && iteration <= numBatches * (epoch + 1)) {
        val input = data.next()
        netD.zeroGradParameters()
        // clamp parameters to a cube
        parametersD.apply1{v =>
          if (v < -0.01) {
            -0.01f
          } else if (v > 0.01) {
            0.01f
          } else {
            v
          }
        }

        // train with real
        val real = input.getInput()
        val errDReal = netD.forward(real).toTensor.value()
        netD.backward(real, ones)

        // train with fake
        noise.apply1(_ => RandomGenerator.RNG.normal(0, 1).toFloat)
        val fake = netG.forward(noise)
        val errDFake = netD.forward(fake).toTensor.value()
        netD.backward(fake, mones)

        val errD = errDReal - errDFake
        println(s"[$epoch/$numEpoch] [$iteration] errDReal is $errDReal, errDFake is $errDFake, errD is $errD")

        // update parameter
        optimizerD.optimize(_ => (errD, gradParametersD), parametersD)

        j += 1
        iteration += 1
      }

      // update Generator network.
      noise.apply1(_ => RandomGenerator.RNG.normal(0, 1).toFloat)
      netG.zeroGradParameters()
      val fake = netG.forward(noise)
      val errG = netD.forward(fake).toTensor.value()
      println(s"[$epoch/$numEpoch] [$gIters] errG is $errG")
      netG.backward(noise, netD.updateGradInput(fake, ones))

      optimizerG.optimize(_ => (errG, gradParametersG), parametersG)
      gIters += 1

      if (iteration > numBatches * (epoch + 1)) {
        dataset.shuffle()
        data = dataset.toLocal().data(true)
        epoch += 1
      }

      if (gIters % 10 == 0) {
        netG.evaluate()
        val gen = netG.forward(fixedNoise).toTensor
        Utils.save(gen, s"/tmp/mlp/fake_${gIters}.jpg")
        netD.saveModule(s"/tmp/mlp/netD_${gIters}.obj", overWrite = true)
        netG.saveModule(s"/tmp/mlp/netG_${gIters}.obj", overWrite = true)
        netG.training()

      }
    }



  }

}
