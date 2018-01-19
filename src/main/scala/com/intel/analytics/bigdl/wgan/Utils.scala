package com.intel.analytics.bigdl.wgan

import java.nio.file.{Path, Paths}

import com.intel.analytics.bigdl.dataset.ByteRecord
import com.intel.analytics.bigdl.dataset.image.{LocalImageFiles, LocalImgReader}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import org.opencv.core.{CvType, Mat}
import org.opencv.imgcodecs.Imgcodecs
import scopt.OptionParser

import scala.collection.mutable.ArrayBuffer

object Utils {
  def save(tensor: Tensor[Float], fileName: String) = {

    val image = Tensor[Float](3, 530, 530).fill(0)
    (0 to 7).foreach{ i =>
      (0 to 7).foreach{ j =>
        image.narrow(2, i * 66 + 3, 64).narrow(3, j * 66 + 3, 64)
          .copy(tensor.select(1, i + j * 8 + 1)).mul(127.5f).add(127.5f)
      }
    }

    val mat = fromTensor(image, format = "CHW")

    Imgcodecs.imwrite(fileName, mat)

  }

  def fromTensor(tensor: Tensor[Float], format: String = "HWC"): OpenCVMat = {
    require(format == "HWC" || format == "CHW", "the format should be HWC or CHW")
    var image = if (format == "CHW") {
      tensor.transpose(1, 2).transpose(2, 3)
    } else {
      tensor
    }
    image = image.contiguous()
    val offset = tensor.storageOffset() - 1
    var floatArr = image.storage().array()
    if (offset > 0) {
      floatArr = floatArr.slice(offset, tensor.nElement() + offset)
    }
    fromFloats(floatArr, image.size(1), image.size(2))
  }

  def fromFloats(floats: Array[Float], height: Int, width: Int): OpenCVMat = {
    var mat: Mat = null
    try {
      mat = new Mat(height, width, CvType.CV_32FC3)
      mat.put(0, 0, floats)
      new OpenCVMat(mat)
    } finally {
      if (null != mat) mat.release()
    }
  }

  case class TrainParams(
      nCpu: Int = 4,
      imageSize: Int = 64,
      nc: Int = 3,
      nz: Int = 100,
      ngf: Int = 512,
      ndf: Int = 64,
      batchSize: Int = 64,
      maxEpoch: Int = 2000,
      learningRateG: Double = 0.00005,
      learningRateD: Double = 0.00005,
      folder: String = "/home/xin/datasets/cifar-10/train"
                        )

  val trainParser = new OptionParser[TrainParams]("BigDL WGAN Example") {
    opt[String]('f', "folder")
      .text("data path")
      .action((x, c) => c.copy(folder = x))
      .required()
    opt[Int]('e', "maxEpoch")
      .text("epoch numbers")
      .action((x, c) => c.copy(maxEpoch = x))
    opt[Double]("learningRateG")
      .text("learning rate of Generator")
      .action((x, c) => c.copy(learningRateG = x))
    opt[Double]("learningRateD")
      .text("learning rate of Discriminator")
      .action((x, c) => c.copy(learningRateD = x))
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
    opt[Int]("nCPU")
      .text("number of CPU to run")
      .action((x, c) => c.copy(nCpu = x))
    opt[Int]("nc")
      .text("input image channels")
      .action((x, c) => c.copy(nCpu = x))
    opt[Int]("nz")
      .text("size of the latent z vector")
      .action((x, c) => c.copy(nCpu = x))
    opt[Int]("ngf")
      .text("ngf")
      .action((x, c) => c.copy(nCpu = x))
    opt[Int]("ndf")
      .text("ndf")
      .action((x, c) => c.copy(nCpu = x))

  }

}
