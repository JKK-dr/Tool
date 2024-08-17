// 用于生成行车隧道
bool TrajectoryOptimizer::GenerateBox(double time, double &x, double &y, double radius, AABox2d &result) const {
  double ri = radius;
  AABox2d bound({-ri, -ri}, {ri, ri});
  if (CheckCollision(time, x, y, bound)) {
    // initial condition not satisfied, involute to find feasible box
    int inc = 4;
    double real_x, real_y;

    do {
      int iter = inc / 4;
      uint8_t edge = inc % 4;

      real_x = x;
      real_y = y;
      if (edge == 0) {
        real_x = x - iter * 0.05;
      } else if (edge == 1) {
        real_x = x + iter * 0.05;
      } else if (edge == 2) {
        real_y = y - iter * 0.05;
      } else if (edge == 3) {
        real_y = y + iter * 0.05;
      }

      inc++;
    } while (CheckCollision(time, real_x, real_y, bound) && inc < config_.corridor_max_iter);
    if (inc > config_.corridor_max_iter) {
      return false;
    }

    x = real_x;
    y = real_y;
  }

  int inc = 4;
  std::bitset<4> blocked;
  double incremental[4] = {0.0};
  double step = radius * 0.2;

  do {
    int iter = inc / 4;
    uint8_t edge = inc % 4;
    inc++;

    if (blocked[edge]) continue;

    incremental[edge] = iter * step;

    AABox2d test({-ri - incremental[0], -ri - incremental[2]},
                 {ri + incremental[1], ri + incremental[3]});

    if (CheckCollision(time, x, y, test) || incremental[edge] >= config_.corridor_incremental_limit) {
      incremental[edge] -= step;
      blocked[edge] = true;
    }
  } while (!blocked.all() && inc < config_.corridor_max_iter);
  if (inc > config_.corridor_max_iter) {
    return false;
  }

  result = {{x - incremental[0], y - incremental[2]},
            {x + incremental[1], y + incremental[3]}};
  return true;
}
