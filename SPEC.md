# PROJECT SPEC — WIND BALANCE SIM (Multiprocess NN Training)

## 0. Goal (Non-Negotiable)

Build a **simple 2D physics simulation** where:

* A **ball** rests on top of a **vertical rectangular stick**
* The stick rests on a **horizontal movable platform**
* **Gravity + wind (Perlin-driven)** perturb the system
* The platform moves left/right to keep the ball balanced
* A **neural network** trains via **multiprocessing** across **X parallel simulations**

This is **not** a realism engine.
This is a **stable, learnable, deterministic toy system**.

---

## 1. Core Constraints (Keep It Simple)

* **Fixed timestep only** (`dt = 1/60`)
* **No rotation physics solver**
* **No impulses**
* **No friction models**
* **No joints**
* **Axis-aligned rectangles only**
* Stick does **NOT rotate**
* Stick can **translate only** (x, y)
* Platform can **translate only in x**
* Wind affects stick + ball
* Platform ignores wind

Balancing emerges because:

* Stick has **finite width**
* Ball must stay within top bounds
* Wind pushes stick sideways
* Platform must compensate

---

## 2. World Geometry

Screen:

* Width: `W`
* Height: `H`
* Hard bounds — **nothing leaves screen**

### Objects

1. **Platform**

   * Rectangle
   * Width: `Pw`
   * Height: `Ph`
   * Position: `(px, py)` (fixed y)
   * Velocity: `(vx, 0)`
   * Controlled by agent
   * Collides with stick + ball
   * Ignores wind

2. **Stick**

   * Rectangle
   * Width: `Sw`
   * Height: `Sh`
   * Position: `(sx, sy)`
   * Velocity: `(vx, vy)`
   * Gravity + wind applied
   * Always **axis-aligned vertical**
   * Must sit on platform top

3. **Ball**

   * Circle
   * Radius: `r`
   * Position: `(bx, by)`
   * Velocity: `(vx, vy)`
   * Gravity + wind applied
   * Collides with stick + platform

---

## 3. Physics Rules (Deliberately Dumb)

### Gravity

```
vy += g * dt
```

### Wind

* Wind is a **horizontal force only**
* Driven by Perlin noise
* Smooth over space + time

```
wind_x = perlin(x * sx, y * sy, t * st) * wind_strength
vx += wind_x * dt
```

Wind affects:

* Stick
* Ball

Wind does NOT affect:

* Platform

---

## 4. Collision Rules (Minimal & Deterministic)

### Ball ↔ Stick (TOP ONLY)

Ball is considered supported if:

```
ball_bottom_y >= stick_top_y
AND
ball_x ∈ [stick_left, stick_right]
```

If true:

* Set `ball_y = stick_top_y + r`
* Set `ball_vy = stick_vy`
* Inherit horizontal velocity partially:

```
ball_vx += stick_vx * 0.5
```

If false:

* Ball falls freely

### Stick ↔ Platform

Stick bottom **always clamps** to platform top:

```
stick_y = platform_top_y + stick_height / 2
stick_vy = 0
```

Stick can slide horizontally.

### Ball ↔ Platform

Same logic as stick top, but platform width.

---

## 5. World Bounds (Hard Clamp)

After every update:

```
x = clamp(x, 0, W)
y = clamp(y, 0, H)

if clamped:
  velocity *= -0.2   # mild damping
```

---

## 6. Platform Control

Platform moves via agent action.

Action space (pick ONE):

* **Discrete**: `{-1, 0, +1}`
* **Continuous**: `vx ∈ [-Vmax, Vmax]`

Movement:

```
platform_x += platform_vx * dt
```

---

## 7. Environment API (OOP — REQUIRED)

### `class BalanceEnv`

Each instance is **fully isolated**.

Required methods:

```python
reset() -> observation
step(action) -> observation, reward, done, info
```

Must contain:

* Platform
* Stick
* Ball
* WindField
* Time accumulator

---

## 8. Observation Vector (Flat, Fixed Size)

Example (engineers may reorder but must document):

```
platform_x / W
platform_vx / Vmax

stick_x / W
stick_vx / Vmax

ball_x / W
ball_y / H
ball_vx / Vmax
ball_vy / Vmax

wind_at_ball / Wmax
wind_at_stick / Wmax
```

---

## 9. Reward Function (Simple, Dense)

Reward **per timestep**:

```
+1.0 if ball is supported on stick
-1.0 if ball falls below platform
```

Optional shaping (allowed):

* small penalty for platform speed
* small penalty for stick drifting too far from center

Episode ends when:

* Ball falls
* Max steps reached

---

## 10. Neural Network (Minimal)

* Feedforward MLP
* 2–3 hidden layers
* Tanh or ReLU
* Outputs platform action

No RNN needed.

---

## 11. Multiprocessing Training (MANDATORY)

### Architecture

* **Main process**

  * Owns NN
  * Updates weights
* **Worker processes**

  * Each runs `K` environments
  * Headless (no rendering)
  * Returns rollouts

### Flow

1. Spawn `N` workers
2. Each worker:

   * Runs envs for `T` steps
   * Collects `(obs, action, reward, done)`
3. Send rollouts to main
4. Main updates NN (PPO or A2C)
5. Broadcast updated weights

### Adjustable Parameters

* `num_workers`
* `envs_per_worker`
* `wind_strength`
* `episode_length`

---

## 12. Rendering (Optional, Debug Only)

* Only one env rendered
* Training NEVER renders
* Renderer subscribes to env state

---

## 13. Acceptance Tests (Engineers MUST PASS)

* [ ] Ball never tunnels through stick
* [ ] With wind = 0, system balances indefinitely
* [ ] Increasing wind causes failure without control
* [ ] Agent learns to survive >10× longer than random
* [ ] Can run overnight without memory leaks
* [ ] Can spawn 100+ envs without slowdown

---

## 14. Explicit Non-Goals

* ❌ Realistic rigid body physics
* ❌ Rotation
* ❌ Friction coefficients
* ❌ Continuous collision detection
* ❌ Fancy constraints
* ❌ Keyboard control for training

---

## Final Note to Engineers

This is **a learning system**, not a physics thesis.

If the sim:

* Is stable
* Is deterministic
* Produces smooth gradients
* Trains overnight

Then it is **correct**, even if it is “physically wrong.”
