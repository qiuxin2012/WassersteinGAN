package com.intel.analytics.bigdl.wgan

import com.intel.analytics.bigdl.nn.AbsCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class WganSpec extends FlatSpec with BeforeAndAfter with Matchers {
  "A Wgan model" should "generate correct output and grad" in {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
    val seed = 100

    val input = Tensor[Float](4, 100, 1, 1).apply1(_ => RandomGenerator.RNG.normal(0, 1).toFloat)

    val code = "torch.manualSeed(" + seed + ")\n" +
      """
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
        |
        |parametersD, gradParametersD = netD:getParameters()
        |parametersG, gradParametersG = netG:getParameters()
        |
        |outputG = netG:forward(input)
        |outputD = netD:forward(outputG)
        |
        |lout = netD.modules[2].output
        |
      """.stripMargin

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input),
      Array("outputG", "outputD", "parametersD", "parametersG" , "netD", "lout"))
    val luaOutputG = torchResult("outputG").asInstanceOf[Tensor[Float]]
    val luaOutputD = torchResult("outputD").asInstanceOf[Tensor[Float]]
    val luaParameterG = torchResult("parametersG").asInstanceOf[Tensor[Float]]
    val luaParameterD = torchResult("parametersD").asInstanceOf[Tensor[Float]]
    val lualout = torchResult("lout").asInstanceOf[Tensor[Float]]
    val luaNetD = torchResult("netD")

    val netG = models.dcganG(64, 100, 3, 64)
    val netD = models.dcganD(64, 100, 3, 64)

    val (parametersD, gradParametersD) = netD.getParameters()
    val (parametersG, gradParametersG) = netG.getParameters()
    parametersD.copy(luaParameterD)
    parametersG.copy(luaParameterG)

    val outputG = netG.forward(input)
    val outputD = netD.forward(outputG)

    luaOutputG should be(outputG)
    luaOutputD.toTensor[Float].mean() should be(outputD)
  }

}
