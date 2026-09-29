"""Section 1.1 - Verify communication with controller target 1."""

import asyncio

import moteus


MOTOR_ID = 1


async def main():
    controller = moteus.Controller(id=MOTOR_ID)
    result = await controller.query()
    print("Controller response:", result)
    print("Position [rev]:", result.values[moteus.Register.POSITION])


if __name__ == "__main__":
    asyncio.run(main())

