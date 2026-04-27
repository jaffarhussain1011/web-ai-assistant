-- ─────────────────────────────────────────────────────────────────────────────
-- sample_data.sql
-- Example MySQL database to test the knowledge-base system.
-- Run with:  mysql -u root -p < data/sample_data.sql
-- ─────────────────────────────────────────────────────────────────────────────

CREATE DATABASE IF NOT EXISTS knowledge_demo
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE knowledge_demo;

-- ── Users ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
  id          INT          UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  name        VARCHAR(120) NOT NULL,
  email       VARCHAR(255) NOT NULL UNIQUE,
  role        ENUM('admin','editor','viewer') NOT NULL DEFAULT 'viewer',
  created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  is_active   TINYINT(1)   NOT NULL DEFAULT 1
);

INSERT INTO users (name, email, role) VALUES
  ('Alice Johnson',  'alice@example.com',  'admin'),
  ('Bob Smith',      'bob@example.com',    'editor'),
  ('Carol Williams', 'carol@example.com',  'editor'),
  ('Dave Brown',     'dave@example.com',   'viewer'),
  ('Eve Davis',      'eve@example.com',    'viewer');

-- ── Products ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS categories (
  id    INT         UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  name  VARCHAR(80) NOT NULL UNIQUE,
  slug  VARCHAR(80) NOT NULL UNIQUE
);

INSERT INTO categories (name, slug) VALUES
  ('Electronics',  'electronics'),
  ('Books',        'books'),
  ('Clothing',     'clothing'),
  ('Home & Garden','home-garden');

CREATE TABLE IF NOT EXISTS products (
  id          INT            UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  category_id INT            UNSIGNED NOT NULL,
  name        VARCHAR(200)   NOT NULL,
  description TEXT,
  price       DECIMAL(10,2)  NOT NULL,
  stock       INT            UNSIGNED NOT NULL DEFAULT 0,
  created_at  DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_product_category
    FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE RESTRICT
);

INSERT INTO products (category_id, name, description, price, stock) VALUES
  (1, 'Wireless Keyboard',    'Compact Bluetooth keyboard with backlight',         49.99, 150),
  (1, 'USB-C Hub 7-in-1',     '7 ports: HDMI, USB-A x3, SD, PD, Ethernet',       34.99,  80),
  (1, 'Noise-Cancelling Headphones', 'Over-ear, 40h battery, foldable',          129.99,  45),
  (2, 'Clean Code',           'Robert C. Martin — software craftsmanship guide',  35.00, 200),
  (2, 'The Pragmatic Programmer', 'Andrew Hunt & David Thomas',                   40.00, 180),
  (3, 'Merino Wool T-Shirt',  '100% merino, anti-odour, machine washable',        55.00,  95),
  (4, 'Bamboo Cutting Board', 'Extra-large, juice groove, end-grain',             29.99,  60);

-- ── Orders ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orders (
  id          INT      UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  user_id     INT      UNSIGNED NOT NULL,
  status      ENUM('pending','processing','shipped','delivered','cancelled')
              NOT NULL DEFAULT 'pending',
  total       DECIMAL(10,2) NOT NULL DEFAULT 0.00,
  created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_order_user
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS order_items (
  id          INT           UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  order_id    INT           UNSIGNED NOT NULL,
  product_id  INT           UNSIGNED NOT NULL,
  quantity    SMALLINT      UNSIGNED NOT NULL DEFAULT 1,
  unit_price  DECIMAL(10,2) NOT NULL,
  CONSTRAINT fk_item_order
    FOREIGN KEY (order_id)   REFERENCES orders(id)   ON DELETE CASCADE,
  CONSTRAINT fk_item_product
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE RESTRICT
);

-- Alice orders a keyboard + headphones
INSERT INTO orders (user_id, status, total) VALUES (1, 'delivered', 179.98);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
  (1, 1, 1, 49.99),
  (1, 3, 1, 129.99);

-- Bob orders two books
INSERT INTO orders (user_id, status, total) VALUES (2, 'shipped', 75.00);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
  (2, 4, 1, 35.00),
  (2, 5, 1, 40.00);

-- Carol orders a hub
INSERT INTO orders (user_id, status, total) VALUES (3, 'processing', 34.99);
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
  (3, 2, 1, 34.99);

-- ── Reviews ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS reviews (
  id          INT           UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  product_id  INT           UNSIGNED NOT NULL,
  user_id     INT           UNSIGNED NOT NULL,
  rating      TINYINT       UNSIGNED NOT NULL CHECK (rating BETWEEN 1 AND 5),
  comment     TEXT,
  created_at  DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_review_product
    FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE,
  CONSTRAINT fk_review_user
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

INSERT INTO reviews (product_id, user_id, rating, comment) VALUES
  (1, 1, 5, 'Great keyboard, very responsive and compact.'),
  (3, 1, 4, 'Excellent noise cancellation, battery life is impressive.'),
  (4, 2, 5, 'A must-read for any developer. Changed how I write code.'),
  (2, 3, 4, 'Works perfectly with my laptop. HDMI output is crisp.');
