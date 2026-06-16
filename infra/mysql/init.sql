-- MySQL bootstrap for the rca-server persistence layer.
-- The `rca` database + `rca` user are also created via MYSQL_DATABASE/MYSQL_USER
-- env; this grants explicitly and is a placeholder for any server-side seed.
CREATE DATABASE IF NOT EXISTS rca CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
GRANT ALL PRIVILEGES ON rca.* TO 'rca'@'%';
FLUSH PRIVILEGES;
