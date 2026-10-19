# Learning-to-Play-Hearthstone-Battlegrounds-Through-Self-Play-Reinforcement-Learning

## Desktop client

Play as a selected seat against random bots, or use spectator mode to watch an
all-bot match:

```powershell
python -m hsbg.play --seed 7
python -m hsbg.play --spectate --seed 7
```

The client uses the shared match engine and random bots. It labels the shop,
hand, and board with minion stats, explains the available actions, and records
plain-language recruit and combat events. It supports pausing, single-action
stepping, playback speed adjustment, and replaying the most recently completed
match. Week 1 intentionally does not include reinforcement learning or
performance-evaluation tools; the current game-rule values remain configurable
prototype assumptions rather than confirmed historical patch rules.

### Basic controls

1. **Buy** minions from the shop; buying spends gold and puts the minion in hand.
2. **Play** minions from hand onto your board. The leftmost board minion attacks
   first.
3. **Sell** a board minion to get the configured gold refund. You can also
   reroll the shop, freeze offers for next turn, or upgrade your Tavern.
4. Press **End Recruit** when ready. Combat starts after all players are ready.
   The event feed explains attacks, shields, deaths, summons, and damage.