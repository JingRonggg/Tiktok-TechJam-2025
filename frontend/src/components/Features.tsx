import { Shield, Zap, Download, Lock, Eye, Cpu } from "lucide-react";
import { Card } from "@/components/ui/card";

const features = [
  {
    icon: Shield,
    title: "AI-Powered Masking",
    description:
      "Advanced machine learning algorithms automatically detect and mask sensitive content including faces, text, and personal information.",
    color: "text-trust-blue",
  },
  {
    icon: Zap,
    title: "Real-time Processing",
    description:
      "Lightning-fast processing ensures your files are protected within seconds, not minutes.",
    color: "text-warning-amber",
  },
  {
    icon: Lock,
    title: "End-to-End Security",
    description:
      "Your files are encrypted during upload, processing, and storage. We never see your original content.",
    color: "text-secure-green",
  },
  {
    icon: Eye,
    title: "Privacy First",
    description:
      "Built with privacy as the foundation. No tracking, no data mining, just pure protection.",
    color: "text-trust-blue",
  },
  {
    icon: Download,
    title: "Easy Download",
    description:
      "Download your protected files in the same format with maintained quality and resolution.",
    color: "text-secure-green",
  },
  {
    icon: Cpu,
    title: "Smart Detection",
    description:
      "Our AI recognizes faces, license plates, documents, and other sensitive elements automatically.",
    color: "text-warning-amber",
  },
];

const Features = () => {
  return (
    <section id="features" className="py-20 bg-surface-elevation/30">
      <div className="container">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
            Advanced Protection Features
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Cutting-edge AI technology meets enterprise-grade security to keep
            your digital content safe.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <Card
              key={index}
              className="bg-gradient-card shadow-soft hover:shadow-strong transition-all duration-300 group animate-fade-in"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="p-8">
                <div
                  className={`w-14 h-14 rounded-xl bg-surface-elevation flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}
                >
                  <feature.icon className={`h-7 w-7 ${feature.color}`} />
                </div>

                <h3 className="text-xl font-semibold text-foreground mb-3">
                  {feature.title}
                </h3>

                <p className="text-muted-foreground leading-relaxed">
                  {feature.description}
                </p>
              </div>
            </Card>
          ))}
        </div>

        <div className="mt-16 text-center">
          <div className="bg-gradient-card rounded-2xl p-8 shadow-strong max-w-4xl mx-auto">
            <div className="grid md:grid-cols-3 gap-8 text-center">
              <div>
                <div className="text-3xl font-bold text-trust-blue mb-2">
                  99.9%
                </div>
                <p className="text-muted-foreground">Detection Accuracy</p>
              </div>
              <div>
                <div className="text-3xl font-bold text-secure-green mb-2">
                  &lt;30s
                </div>
                <p className="text-muted-foreground">Average Processing Time</p>
              </div>
              <div>
                <div className="text-3xl font-bold text-warning-amber mb-2">
                  256-bit
                </div>
                <p className="text-muted-foreground">Encryption Standard</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Features;
