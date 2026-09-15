from dataclasses import dataclass
from enum import Enum

class CHANNEL_ESTIMATOR(Enum):
    LS = "LS"

class EQUALIZER(Enum):
    ZF = "ZF"

class CHANNEL_MODEL(Enum):
    Rayleigh = "Rayleigh"

@dataclass
class CarrierConfig:
    nPRB: int
    subcarrierSpacing: float
    nOFDMSymbolsPerSlot: int
    NFFT: int

@dataclass
class BWPConfig:
    allocatedPRB: list[int]

@dataclass
class PDSCHConfig:
    allocatedSymbols: list[int]
    Qm: int
    R: float
    nLayer: int
    slotNumInFrame: int

@dataclass
class DMRSConfig:
    allocatedDMRSPerPRB: list[tuple[int, int]]
    N_DMRS_ID: int
    lambda_bar: int
    n_SCID: int

@dataclass
class ScramblingConfig:
    nRNTI: int
    nID: int

@dataclass
class HARQConfig:
    rv_id: int

@dataclass
class OFDMConfig:
    NFFT: int

@dataclass
class ChannelConfig:
    model: CHANNEL_MODEL
    velocity: float
    carrierFrequency: float
    delays_ns: list[float]
    delayPower_dB: list[float]

@dataclass
class ReceiverConfig:
    maxLDPCIterations: int
    equalizerType: EQUALIZER
    channelEstimatorType: CHANNEL_ESTIMATOR

@dataclass
class NRSystemConfig:
    carrier: CarrierConfig
    bwp: BWPConfig
    pdsch: PDSCHConfig
    dmrs: DMRSConfig
    scrambling: ScramblingConfig
    harq: HARQConfig
    channel: ChannelConfig
    receiver: ReceiverConfig
        

carrierConfig = CarrierConfig(
    nPRB=50,
    subcarrierSpacing=30e3,
    nOFDMSymbolsPerSlot=14,
    NFFT=1024,
)

bwpConfig = BWPConfig(
    allocatedPRB=list(range(5, 15))
)

pdschConfig = PDSCHConfig(
    allocatedSymbols=list(range(2, 14)),
    Qm=4,
    R=0.3,
    nLayer=1,
    slotNumInFrame=0,
)

dmrsConfig = DMRSConfig(
    allocatedDMRSPerPRB=[
        (0,2),
        (2,2),
        (4,2),
        (6,2),
        (8,2),
        (10,2)
    ],
    N_DMRS_ID=100,
    lambda_bar=0,
    n_SCID=0,
)

scramblingConfig = ScramblingConfig(
    nRNTI=99,
    nID=42,
)

harqConfig = HARQConfig(
    rv_id=0
)

channelConfig = ChannelConfig(
    model=CHANNEL_MODEL.Rayleigh,
    velocity=15,
    carrierFrequency=2.5e9,
    delays_ns=[0, 100, 300],
    delayPower_dB=[0, -3, -6]
)

receiverConfig = ReceiverConfig(
    maxLDPCIterations=10,
    equalizerType=EQUALIZER.ZF,
    channelEstimatorType=CHANNEL_ESTIMATOR.LS,
)
    