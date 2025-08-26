import { Shield, Lock, Eye } from "lucide-react";
import { Button } from "@/components/ui/button";
import heroImage from "@/assets/hero-privacy.jpg";

const Hero = () => {
  return (
    <section className="relative min-h-[80vh] flex items-center bg-gradient-hero">
      <div className="absolute inset-0 bg-gradient-to-r from-background via-background/80 to-transparent z-10" />
      <div
        className="absolute inset-0 bg-cover bg-center bg-no-repeat opacity-20"
        style={{ backgroundImage: `url(${heroImage})` }}
      />

      <div className="container relative z-20 grid lg:grid-cols-2 gap-12 items-center">
        <div className="space-y-8 animate-fade-in">
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-trust-blue">
              <Shield className="h-5 w-5" />
              <span className="text-sm font-medium uppercase tracking-wider">
                AI-Powered Privacy Protection
              </span>
            </div>
            <h1 className="text-4xl md:text-6xl font-bold text-foreground leading-tight">
              Protect Your
              <span className="text-trust-blue"> Digital Identity</span>
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl">
              Advanced AI technology that automatically masks sensitive content
              in your images and videos, ensuring your privacy while maintaining
              visual quality.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row gap-4">
            <Button variant="hero" size="lg" className="text-lg">
              Start Protecting Now
            </Button>
            <Button variant="outline" size="lg" className="text-lg">
              Learn More
            </Button>
          </div>

          <div className="grid grid-cols-3 gap-6 pt-8">
            <div className="text-center">
              <div className="bg-surface-elevation rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-2">
                <Lock className="h-6 w-6 text-trust-blue" />
              </div>
              <p className="text-sm font-medium">Secure Upload</p>
            </div>
            <div className="text-center">
              <div className="bg-surface-elevation rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-2">
                <Shield className="h-6 w-6 text-secure-green" />
              </div>
              <p className="text-sm font-medium">AI Protection</p>
            </div>
            <div className="text-center">
              <div className="bg-surface-elevation rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-2">
                <Eye className="h-6 w-6 text-warning-amber" />
              </div>
              <p className="text-sm font-medium">Privacy First</p>
            </div>
          </div>
        </div>

        <div className="hidden lg:block">
          <div className="bg-gradient-card rounded-2xl p-8 shadow-strong">
            <div className="space-y-6">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 bg-secure-green rounded-full animate-pulse-glow"></div>
                <span className="text-sm font-medium text-muted-foreground">
                  AI Protection Active
                </span>
              </div>
              <div className="space-y-4">
                <div className="bg-surface-elevation rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Face Detection</span>
                    <span className="text-xs text-secure-green">
                      ✓ Protected
                    </span>
                  </div>
                  <div className="w-full bg-secondary rounded-full h-2">
                    <div className="bg-secure-green h-2 rounded-full w-full"></div>
                  </div>
                </div>
                <div className="bg-surface-elevation rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">
                      Text Recognition
                    </span>
                    <span className="text-xs text-secure-green">✓ Masked</span>
                  </div>
                  <div className="w-full bg-secondary rounded-full h-2">
                    <div className="bg-secure-green h-2 rounded-full w-full"></div>
                  </div>
                </div>
                <div className="bg-surface-elevation rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">
                      Object Detection
                    </span>
                    <span className="text-xs text-trust-blue">
                      Processing...
                    </span>
                  </div>
                  <div className="w-full bg-secondary rounded-full h-2">
                    <div className="bg-trust-blue h-2 rounded-full w-3/4 animate-pulse"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
