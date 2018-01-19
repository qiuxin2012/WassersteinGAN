package com.intel.analytics.bigdl.wgan

import com.intel.analytics.bigdl.nn.AbsCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class WganSpec extends FlatSpec with BeforeAndAfter with Matchers {
  "A Wgan model" should "generate correct output and grad" in {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
    val criterion = new AbsCriterion[Double]()

    val input = Tensor[Double](3)
    input(Array(1)) = 0.4
    input(Array(2)) = 0.5
    input(Array(3)) = 0.6

    val target = Tensor[Double](3)
    target(Array(1)) = 0
    target(Array(2)) = 1
    target(Array(3)) = 1

    val start = System.nanoTime()
    val output1 = criterion.forward(input, target)
    val output2 = criterion.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "abs = nn.AbsCriterion()\n" +
      "output1 = abs:forward(input, target)\n " +
      "output2 = abs:backward(input, target)"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "target" -> target),
      Array("output1", "output2"))
    val luaOutput1 = torchResult("output1").asInstanceOf[Double]
    val luaOutput2 = torchResult("output2").asInstanceOf[Tensor[Double]]

    luaOutput1 should be(output1)
    luaOutput2 should be(output2)

    println("Test case : AbsCriterion, Torch : " + luaTime + " s, Scala : " +
      scalaTime / 1e9 + " s")
  }

}
