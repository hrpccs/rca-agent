---
kind: domain_fact
---

# OpenTelemetry Demo service topology (rca100 benchmark)

The rca100 benchmark cases are built on the
[OpenTelemetry Demo](https://github.com/open-telemetry/opentelemetry-demo)
microservices application. Knowing the canonical service set and its
dependencies speeds up investigation: when an alert names one service, the
candidate root-cause locations are its direct dependencies.

## Services

The benchmark topology (e.g. case `t001`) contains these `apm.service`
entities, matching the otel-demo component list:

`accounting`, `ad`, `cart`, `checkout`, `currency`, `email`, `flagd`,
`fraud-detection`, `frontend`, `frontend-proxy`, `frontend-web`,
`image-provider`, `inventory`, `load-generator`, `payment`, `product-catalog`,
`quote`, `recommendation`, `shipping`.

## Roles

| Service | Role |
|---|---|
| `frontend` / `frontend-web` | user-facing web app; fans out to cart, checkout, product-catalog, recommendation, ad, currency |
| `frontend-proxy` | edge proxy (Envoy); routes to frontend, image-provider, flagd |
| `checkout` | order orchestration; the most-connected service — calls cart, currency, email, payment, product-catalog, shipping, flagd, and emits to the `orders` message queue |
| `cart` | shopping cart; calls inventory (and the cart RPC methods `GetCart`/`EmptyCart`) |
| `product-catalog` | product catalog; called by frontend and checkout (`GetProduct`) |
| `currency` | currency conversion (`Convert`); called by frontend and checkout |
| `payment` | payment charging (`Charge`); called by checkout |
| `email` | order confirmation email; called by checkout |
| `shipping` | shipping; calls quote (`GetQuote`/`ShipOrder`) |
| `quote` | shipping quote (internal) |
| `recommendation` | product recommendations; called by frontend |
| `ad` | ad placement; called by frontend |
| `inventory` | stock; **depends on the external MySQL RDS** (`rm-…mysql…aliyuncs.com:3306`) |
| `flagd` | feature-flag service (`flagd:8015`); called by checkout and others |
| `fraud-detection` | fraud check (async) |
| `accounting` | order accounting (async) |
| `image-provider` | product images; served via frontend-proxy |
| `load-generator` | synthetic traffic generator |

## Key dependency chains

- **checkout is the hub.** It calls cart, currency, payment, email,
  product-catalog, shipping, and flagd, and publishes to the `orders` message
  topic. An alert on `checkout::/oteldemo.CheckoutService/PlaceOrder` means
  the root cause could be any of these — trace the slowest/erroring span to
  localize.
- **cart → inventory → MySQL RDS.** Cart calls inventory over HTTP
  (`inventory:9090`), and inventory is the service backed by the external
  MySQL database. DB problems (slow queries, connection saturation, missing
  index) surface as cart/checkout latency.
- **frontend → checkout / cart / product-catalog / recommendation / ad /
  currency.** Frontend aggregates many services, so a frontend latency alert
  is usually a downstream regression — find the slowest downstream span.
- **flagd is a cross-cutting dependency.** Feature-flag changes (a flag flip,
  a bad flag config) can change behavior service-wide; check flagd when a
  behavior change has no obvious deploy.

## Topology entity types

Beyond `apm.service`, cases include: `apm.operation` (RPC/HTTP operations
like `/oteldemo.CheckoutService/PlaceOrder`), `apm.instance` (service
instances), `apm.external.{database,http_client,rpc_client,message}` (the
externalized endpoints/queues seen above), and the full Kubernetes layer:
`k8s.cluster`, `k8s.namespace` (`cms-demo`), `k8s.node`, `k8s.pod`,
`k8s.service`, `k8s.configmap`, `k8s.ingress`, `k8s.job`/`cronjob`,
`k8s.persistentvolume`, `k8s.storageclass`. Edges are `contains`, `calls`, or
`hosts`.

## Investigation tips

- The alerted operation name usually encodes the service and RPC method
  (e.g. `checkout::/oteldemo.CheckoutService/PlaceOrder`). The service prefix
  is your entry entity.
- Because `checkout` fans out widely, a checkout alert frequently resolves to
  a downstream dependency (cart, currency, payment, email, shipping,
  product-catalog). Trace before theorizing.
- `inventory` is the only service with an external DB dependency; DB-related
  fault types (`db.slow_query`, connection saturation) almost always involve
  inventory or a service calling it (cart, checkout).
