import Navigation from '@/components/Navigation';
import CarScroll from '@/components/CarScroll';
import SpecsGrid from '@/components/SpecsGrid';
import FeatureCards from '@/components/FeatureCards';
import CTASection from '@/components/CTASection';
import Footer from '@/components/Footer';

export default function Home() {
  return (
    <main className="bg-black min-h-screen selection:bg-white/20">
      <Navigation />
      <CarScroll />
      <SpecsGrid />
      <FeatureCards />
      <CTASection />
      <Footer />
    </main>
  );
}
