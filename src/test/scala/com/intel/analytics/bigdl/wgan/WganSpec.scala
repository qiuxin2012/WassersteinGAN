package com.intel.analytics.bigdl.wgan

import com.intel.analytics.bigdl.nn.{AbsCriterion, Module}
import com.intel.analytics.bigdl.optim.{Adam, RMSprop, SGD}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class WganSpec extends FlatSpec with BeforeAndAfter with Matchers {
  "A Wgan model" should "generate correct output and grad" in {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
    val seed = 100
    RandomGenerator.RNG.setSeed(seed)

    val input = Tensor[Float](4, 100, 1, 1).apply1(_ => RandomGenerator.RNG.normal(0, 1).toFloat)

    val code = "torch.manualSeed(" + seed + ")\n" +
      """
        |require 'optim'
        |torch.setdefaulttensortype('torch.FloatTensor')
        |local function weights_init(m)
        |    local name = torch.type(m)
        |    if name:find('Convolution') then
        |        m.weight:normal(0.0, 0.02)
        |        m:noBias()
        |    elseif name:find('BatchNormalization') then
        |        if m.weight then m.weight:normal(1.0, 0.02) end
        |        if m.bias then m.bias:fill(0) end
        |    end
        |end
        |
        |local SpatialBatchNormalization = nn.SpatialBatchNormalization
        |local SpatialConvolution = nn.SpatialConvolution
        |local SpatialFullConvolution = nn.SpatialFullConvolution
        |
        |function get_netG(nz, ngf, nc)
        |    local netG = nn.Sequential()
        |    -- input is Z, going into a convolution
        |    netG:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))
        |    netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
        |    -- state size: (ngf*8) x 4 x 4
        |    netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
        |    netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
        |    -- state size: (ngf*4) x 8 x 8
        |    netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
        |    netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
        |    -- state size: (ngf*2) x 16 x 16
        |    netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
        |    netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
        |    -- state size: (ngf) x 32 x 32
        |    netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
        |    netG:add(nn.Tanh())
        |    -- state size: (nc) x 64 x 64
        |
        |    netG:apply(weights_init)
        |
        |    return netG
        |end
        |
        |function get_netD(nc, ndf)
        |    local netD = nn.Sequential()
        |
        |    -- input is (nc) x 64 x 64
        |    netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
        |    netD:add(nn.LeakyReLU(0.2, true))
        |    -- state size: (ndf) x 32 x 32
        |    netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
        |    netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
        |    -- state size: (ndf*2) x 16 x 16
        |    netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
        |    netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
        |    -- state size: (ndf*4) x 8 x 8
        |    netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
        |    netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
        |    -- state size: (ndf*8) x 4 x 4
        |    netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
        |    -- netD:add(nn.Sigmoid())
        |    -- state size: 1 x 1 x 1
        |    netD:add(nn.View(1):setNumInputDims(3))
        |    -- state size: 1
        |
        |    netD:apply(weights_init)
        |
        |    return netD
        |end
        |
        |nz = 100
        |ngf = 64
        |nc = 3
        |ndf = 64
        |netG = get_netG(nz, ngf, nc)
        |netD = get_netD(nc, ndf)
        |
        |ones = torch.Tensor(4):fill(1)
        |
        |parametersD, gradParametersD = netD:getParameters()
        |parametersG, gradParametersG = netG:getParameters()
        |
        |initParametersD = parametersD:clone()
        |initParametersG = parametersG:clone()
        |
        |optimStateG = {
        |   learningRate = 0.05
        |}
        |
        |fGx = function(x)
        |  gradParametersG:zero()
        |  outputG = netG:forward(input)
        |  outputD = netD:forward(outputG)
        |  err = outputD:mean()
        |
        |  gradInputD = netD:backward(outputG, ones)
        |  gradInputG = netG:backward(input, gradInputD)
        |  return err, gradParametersG
        |end
        |
        |for i = 1,5,1 do
        |  -- optim.sgd(fGx, parametersG, optimStateG)
        |  -- optim.rmsprop(fGx, parametersG, optimStateG)
        |  optim.adam(fGx, parametersG, optimStateG)
        |end
        |
        |lout = netD.modules[2].output
        |
      """.stripMargin

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input),
      Array("outputG", "outputD", "parametersD", "parametersG",
        "netD", "lout", "gradParametersD", "gradParametersG",
        "gradInputD", "gradInputG", "initParametersD", "initParametersG"))
    val luaOutputG = torchResult("outputG").asInstanceOf[Tensor[Float]]
    val luaOutputD = torchResult("outputD").asInstanceOf[Tensor[Float]]
    val luaParameterG = torchResult("parametersG").asInstanceOf[Tensor[Float]]
    val luaParameterD = torchResult("parametersD").asInstanceOf[Tensor[Float]]
    val luaGradParameterG = torchResult("gradParametersG").asInstanceOf[Tensor[Float]]
    val luaGradParameterD = torchResult("gradParametersD").asInstanceOf[Tensor[Float]]
    val luaGradInputG = torchResult("gradInputG").asInstanceOf[Tensor[Float]]
    val luaGradInputD = torchResult("gradInputD").asInstanceOf[Tensor[Float]]
    val luaInitParametersD = torchResult("initParametersD").asInstanceOf[Tensor[Float]]
    val luaInitParametersG = torchResult("initParametersG").asInstanceOf[Tensor[Float]]

    val lualout = torchResult("lout").asInstanceOf[Tensor[Float]]
    val luaNetD = torchResult("netD")

    val netG = models.dcganG(64, 100, 3, 64)
    val netD = models.dcganD(64, 100, 3, 64)

    val (parametersD, gradParametersD) = netD.getParameters()
    val (parametersG, gradParametersG) = netG.getParameters()
    parametersD.copy(luaInitParametersD)
    parametersG.copy(luaInitParametersG)
//    val rmsProp = new RMSprop[Float](0.05)
//    val rmsProp = new SGD[Float](0.05)
    val rmsProp = new Adam[Float](0.05)

    (1 to 5).foreach { i =>
      netG.zeroGradParameters()
      val outputG = netG.forward(input)
      val outputD = netD.forward(outputG)

      val ones = Tensor[Float](1).fill(4)
      val gradInputD = netD.backward(outputG, ones)
      val gradInputG = netG.backward(input, gradInputD)
      rmsProp.optimize(_ => (outputD.toTensor[Float].value(), gradParametersG), parametersG)
    }

    println(parametersG.abs().sum())
    println(luaParameterG.abs().sum())

    println(gradParametersG.abs().sum())
    println(luaGradParameterG.abs().sum())
    // luaOutputG should be(outputG)
    // luaOutputD.toTensor[Float].mean() should be(outputD)
  }

  "A Wgan model" should "generate correct output" in {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
    val seed = 100
    RandomGenerator.RNG.setSeed(seed)

    val input = Tensor[Float](4, 100, 1, 1).apply1(_ => RandomGenerator.RNG.normal(0, 1).toFloat)

    val netG = Module.loadModule[Float]("/tmp/netG_100.obj")
    val netD = Module.loadModule[Float]("/tmp/netD_100.obj")
//    val netG1 = models.dcganG(64, 100, 3, 64)
//    val netD1 = models.dcganD(64, 100, 3, 64)
//    val (parametersD1, gradParametersD1) = netD1.getParameters()
//    val (parametersG1, gradParametersG1) = netG1.getParameters()

    netG.training()
    netD.training()
    val (parametersD, gradParametersD) = netD.getParameters()
    val (parametersG, gradParametersG) = netG.getParameters()

    val code = "torch.manualSeed(" + seed + ")\n" +
      """
        |require 'optim'
        |torch.setdefaulttensortype('torch.FloatTensor')
        |local function weights_init(m)
        |    local name = torch.type(m)
        |    if name:find('Convolution') then
        |        m.weight:normal(0.0, 0.02)
        |        m:noBias()
        |    elseif name:find('BatchNormalization') then
        |        if m.weight then m.weight:normal(1.0, 0.02) end
        |        if m.bias then m.bias:fill(0) end
        |    end
        |end
        |
        |local SpatialBatchNormalization = nn.SpatialBatchNormalization
        |local SpatialConvolution = nn.SpatialConvolution
        |local SpatialFullConvolution = nn.SpatialFullConvolution
        |
        |function get_netG(nz, ngf, nc)
        |    local netG = nn.Sequential()
        |    -- input is Z, going into a convolution
        |    netG:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))
        |    netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
        |    -- state size: (ngf*8) x 4 x 4
        |    netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
        |    netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
        |    -- state size: (ngf*4) x 8 x 8
        |    netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
        |    netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
        |    -- state size: (ngf*2) x 16 x 16
        |    netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
        |    netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
        |    -- state size: (ngf) x 32 x 32
        |    netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
        |    netG:add(nn.Tanh())
        |    -- state size: (nc) x 64 x 64
        |
        |    netG:apply(weights_init)
        |
        |    return netG
        |end
        |
        |function get_netD(nc, ndf)
        |    local netD = nn.Sequential()
        |
        |    -- input is (nc) x 64 x 64
        |    netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
        |    netD:add(nn.LeakyReLU(0.2, true))
        |    -- state size: (ndf) x 32 x 32
        |    netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
        |    netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
        |    -- state size: (ndf*2) x 16 x 16
        |    netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
        |    netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
        |    -- state size: (ndf*4) x 8 x 8
        |    netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
        |    netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
        |    -- state size: (ndf*8) x 4 x 4
        |    netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
        |    -- netD:add(nn.Sigmoid())
        |    -- state size: 1 x 1 x 1
        |    netD:add(nn.View(1):setNumInputDims(3))
        |    -- state size: 1
        |
        |    netD:apply(weights_init)
        |
        |    return netD
        |end
        |
        |nz = 100
        |ngf = 64
        |nc = 3
        |ndf = 64
        |netG = get_netG(nz, ngf, nc)
        |netD = get_netD(nc, ndf)
        |
        |ones = torch.Tensor(4):fill(1)
        |
        |parametersD, gradParametersD = netD:getParameters()
        |parametersG, gradParametersG = netG:getParameters()
        |
        |parametersD:copy(initD)
        |parametersG:copy(initG)
        |
        |outputG = netG:forward(input)
        |outputD = netD:forward(outputG)
        |--err = outputD:mean()
        |
      """.stripMargin

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "initD" -> parametersD,
      "initG" -> parametersG),
      Array("outputG", "outputD"))
    val luaOutputG = torchResult("outputG").asInstanceOf[Tensor[Float]]
    val luaOutputD = torchResult("outputD").asInstanceOf[Tensor[Float]]

    val outputG = netG.forward(input)
    val outputD = netD.forward(outputG)
    println()

  }

}
